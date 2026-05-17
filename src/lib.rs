#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(any(doc, doctest), doc = include_str!("../README.md"))]

//! ---
//! # Documentation map
//!
//! The README above is included verbatim and serves as the **user-facing introduction** to the
//! crate (overview, features, and quick-start examples).
//!
//! Everything below this line specifies the **semantic and correctness contract** of the
//! `delaunay` crate and is intended for users who need stronger guarantees, deeper understanding
//! of invariants, or who are extending the implementation.
//!
//! This crate’s documentation is intentionally layered by audience and intent:
//!
//! - **README.md** (included above):
//!   User-facing overview, feature list, and quick-start examples.
//!
//! - **Crate-level documentation (`lib.rs`)** (this document):
//!   The programming contract of the library: what invariants are enforced, when validation runs,
//!   and what errors mean.
//!
//!   In particular, this document covers:
//!   - The validation hierarchy and invariant stack (Levels 1–4)
//!   - Topological guarantees (`TopologyGuarantee`) and insertion-time validation policy (`ValidationPolicy`)
//!   - High-level error semantics and programming contract (transactional operations, duplicate rejection)
//!
//! - **docs/workflows.md**:
//!   Task-oriented, end-to-end usage recipes (Builder API, Edit API, validation,
//!   repairs, diagnostics, and statistics).
//!
//! - **docs/validation.md**:
//!   Formal definitions of validation Levels 1–4, their costs, and guidance on when
//!   each level should be applied.
//!
//! - **docs/diagnostics.md**:
//!   Opt-in diagnostic helpers, structured reports, debug switches, and guidance for
//!   producing useful failure reports without expanding the default API surface.
//!
//! - **docs/invariants.md**:
//!   Deeper theoretical discussion of topological and geometric invariants
//!   (PL-manifold conditions, ridge/vertex links, ordering heuristics, and
//!   convergence assumptions), plus algorithmic background and limitations.
//!
//! ## Which import do I need?
//!
//! The crate provides several focused prelude modules.  Pick the one that
//! matches your task:
//!
//! | Task | Import |
//! |---|---|
//! | Construct/configure a Delaunay triangulation | `use delaunay::prelude::triangulation::construction::*` |
//! | Low-level incremental insertion building blocks | `use delaunay::prelude::triangulation::insertion::*` |
//! | Read-only queries, traversal, convex hull | `use delaunay::prelude::query::*` |
//! | Point location and conflict-region algorithms | `use delaunay::prelude::algorithms::*` |
//! | Geometry helpers, predicates, points | `use delaunay::prelude::geometry::*` |
//! | Random points / triangulations for examples and tests | `use delaunay::prelude::generators::*` |
//! | Hilbert ordering and quantization utilities | `use delaunay::prelude::ordering::*` |
//! | Bistellar flips (Pachner moves) | `use delaunay::prelude::triangulation::flips::*` |
//! | Delaunay repair and flip-based Level 4 validation | `use delaunay::prelude::triangulation::repair::*` |
//! | Delaunayize workflow (repair + flip) | `use delaunay::prelude::triangulation::delaunayize::*` |
//! | Construction telemetry diagnostics | `use delaunay::prelude::triangulation::diagnostics::*` |
//! | Construction validation cadence/policy | `use delaunay::prelude::triangulation::validation::*` |
//! | Topology validation, Euler characteristic | `use delaunay::prelude::topology::validation::*` |
//! | Topological spaces and topology traits | `use delaunay::prelude::topology::spaces::*` |
//! | Low-level TDS cells, facets, keys | `use delaunay::prelude::tds::*` |
//! | Collection types (`FastHashMap`, etc.) | `use delaunay::prelude::collections::*` |
//! | Legacy broad triangulation import | `use delaunay::prelude::triangulation::*` |
//! | Everything (kitchen sink) | `use delaunay::prelude::*` |
//!
//! ## Public low-level namespace policy
//!
//! The implementation namespace remains `crate::core`, but the public
//! low-level surface is exposed through curated modules:
//! [`tds`](crate::tds), [`collections`](crate::collections),
//! [`algorithms`](crate::algorithms), and [`query`](crate::query), plus the
//! matching focused preludes. These names describe the data structures and
//! workflows users compose without colliding with Rust's standard `core`
//! vocabulary.
//!
//! Prefer these curated modules and focused preludes in examples, doctests,
//! benchmarks, and downstream-style integration tests. High-level Delaunay
//! construction remains outside the low-level TDS/query surface.
//!
//! ## Examples (contract-oriented)
//!
//! ### Validation hierarchy (Levels 1–4)
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
//! };
//! use delaunay::prelude::triangulation::insertion::InsertionError;
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // Levels 1–2: elements + structural (TDS)
//! assert!(dt.tds().validate().is_ok());
//!
//! // Levels 1–3: elements + structural + topology
//! assert!(dt.as_triangulation().validate().is_ok());
//!
//! // Level 4 only: Delaunay property (assumes Levels 1–3)
//! assert!(dt.is_valid().is_ok());
//!
//! // Levels 1–4: full cumulative validation
//! assert!(dt.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ### Topology guarantees and insertion-time validation (`TopologyGuarantee`, `ValidationPolicy`)
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, TopologyGuarantee,
//!     vertex,
//! };
//! use delaunay::prelude::triangulation::validation::ValidationPolicy;
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
//! assert_eq!(dt.validation_policy(), ValidationPolicy::OnSuspicion);
//!
//! dt.set_topology_guarantee(TopologyGuarantee::Pseudomanifold);
//! dt.set_validation_policy(ValidationPolicy::Always);
//!
//! assert_eq!(dt.topology_guarantee(), TopologyGuarantee::Pseudomanifold);
//! assert_eq!(dt.validation_policy(), ValidationPolicy::Always);
//! # Ok(())
//! # }
//! ```
//!
//! ### Transactional operations and duplicate rejection
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
//! };
//! use delaunay::prelude::triangulation::insertion::InsertionError;
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.0, 1.0]),
//! ];
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! let before_vertices = dt.number_of_vertices();
//! let before_cells = dt.number_of_cells();
//!
//! // Duplicate coordinates are rejected.
//! let result = dt.insert(vertex!([0.0, 0.0]));
//! assert!(matches!(result, Err(InsertionError::DuplicateCoordinates { .. })));
//!
//! // On error, the triangulation is unchanged.
//! assert_eq!(dt.number_of_vertices(), before_vertices);
//! assert_eq!(dt.number_of_cells(), before_cells);
//! # Ok(())
//! # }
//! ```
//!
//! # Triangulation invariants and validation hierarchy
//!
//! The crate is organized as a small **validation stack**, where each layer adds additional
//! invariants on top of the preceding one:
//!
//! - [`Vertex`](crate::core::vertex::Vertex) and [`Cell`](crate::core::cell::Cell) provide
//!   **element validity** checks.
//!   Level 1 (elements) validation checks invariants such as:
//!   - **Vertex coordinates** – finite (no NaN/∞) and UUID is non-nil.
//!   - **Cell shape** – exactly D+1 distinct vertex keys, valid UUID, and neighbor buffer length
//!     (if present) is D+1.
//!
//!   These checks are surfaced via [`Vertex::is_valid`](crate::core::vertex::Vertex::is_valid) and
//!   [`Cell::is_valid`](crate::core::cell::Cell::is_valid), and are automatically run by
//!   [`Tds::validate`](crate::core::tds::Tds::validate) (Levels 1–2).
//!
//! - [`Tds`](crate::core::tds::Tds) (Triangulation Data Structure)
//!   stores the **combinatorial / structural** representation.
//!   Level 2 (structural) validation checks invariants such as:
//!   - **Vertex mappings** – every vertex UUID has a corresponding key and vice versa.
//!   - **Cell mappings** – every cell UUID has a corresponding key and vice versa.
//!   - **No duplicate cells** – no two maximal cells share the same vertex set.
//!   - **Facet sharing** – each facet is shared by at most 2 cells (1 on the boundary, 2 in the interior).
//!   - **Neighbor consistency** – neighbor relationships are mutual and reference a shared facet.
//!
//!   These checks are surfaced via [`Tds::is_valid`](crate::core::tds::Tds::is_valid)
//!   (structural only) and [`Tds::validate`](crate::core::tds::Tds::validate)
//!   (Levels 1–2, elements + structural). For cumulative diagnostics across the full stack,
//!   use [`DelaunayTriangulation::validation_report`](triangulation::delaunay::DelaunayTriangulation::validation_report).
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
//! - [`DelaunayTriangulation`](crate::triangulation::delaunay::DelaunayTriangulation) builds on
//!   `Triangulation` and validates the **geometric** Delaunay condition.
//!   Level 4 (Delaunay property) validation is performed by
//!   [`DelaunayTriangulation::is_valid`](triangulation::delaunay::DelaunayTriangulation::is_valid) (Level 4 only) and
//!   [`DelaunayTriangulation::validate`](triangulation::delaunay::DelaunayTriangulation::validate) (Levels 1–4).
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
//! - Level 1 (elements / `Vertex` + `Cell`): `Vertex::is_valid()` / `Cell::is_valid()` for element
//!   checks, or `dt.tds().validate()` for Levels 1–2.
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
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulation, DelaunayTriangulationConstructionError, vertex,
//! };
//! use delaunay::prelude::triangulation::insertion::InsertionError;
//! use delaunay::prelude::triangulation::validation::ValidationPolicy;
//!
//! # #[derive(Debug, thiserror::Error)]
//! # enum ExampleError {
//! #     #[error(transparent)]
//! #     Construction(#[from] DelaunayTriangulationConstructionError),
//! #     #[error(transparent)]
//! #     Insertion(#[from] InsertionError),
//! # }
//! # fn main() -> Result<(), ExampleError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let mut dt = DelaunayTriangulation::new(&vertices)?;
//!
//! // Performance mode: disable insertion-time Level 3 topology validation.
//! dt.set_validation_policy(ValidationPolicy::Never);
//!
//! // Do incremental work...
//! dt.insert(vertex!([0.2, 0.2, 0.2]))?;
//!
//! // ...then explicitly validate the topology layer when you need a certificate.
//! assert!(dt.as_triangulation().validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ### Choosing Level 3 topology guarantee (`TopologyGuarantee`)
//!
//! This section specifies *what* invariants are enforced. The formal topological
//! definitions and rationale live in `docs/invariants.md`.
//!
//! Level 3 topology validation is parameterized by
//! [`TopologyGuarantee`](crate::core::triangulation::TopologyGuarantee). This is separate from
//! `ValidationPolicy`: it controls *what* invariants Level 3 enforces, not *when* automatic
//! validation runs.
//!
//! - [`TopologyGuarantee::PLManifold`](crate::core::triangulation::TopologyGuarantee::PLManifold)
//!   (default): enforces manifold facet degree, boundary closure, connectedness, Euler characteristic,
//!   and link-based manifold conditions. Ridge-link checks are applied incrementally during insertion,
//!   with vertex-link validation performed at construction completion.
//!
//!   The formal topological definitions, link conditions, and rationale for this validation strategy
//!   are documented in `docs/invariants.md`.
//! - [`TopologyGuarantee::PLManifoldStrict`](crate::core::triangulation::TopologyGuarantee::PLManifoldStrict):
//!   vertex-link validation after every insertion (slowest, maximum safety).
//! - [`TopologyGuarantee::Pseudomanifold`](crate::core::triangulation::TopologyGuarantee::Pseudomanifold):
//!   skips vertex-link validation (may be faster), but bistellar flip convergence is not guaranteed and
//!   you may want to validate the Delaunay property explicitly for near-degenerate inputs.
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
//! };
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // For `TopologyGuarantee::PLManifold`, full certification includes a completion-time
//! // vertex-link validation pass.
//! assert!(dt.as_triangulation().validate_at_completion().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
//! };
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0, 0.0]),
//!     vertex!([1.0, 0.0, 0.0]),
//!     vertex!([0.0, 1.0, 0.0]),
//!     vertex!([0.0, 0.0, 1.0]),
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // `validate()` returns the first violation; `validation_report()` is intended for
//! // debugging/telemetry where you want the full set of violated invariants.
//! assert!(dt.validation_report().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! For implementation details on invariant enforcement, see the incremental
//! insertion implementation.
//!
//! # Programming contract (high-level)
//!
//! - **Transactional mutations**: Construction and incremental operations are designed to be
//!   all-or-nothing. If an operation returns `Err(_)`, the triangulation is rolled back to its
//!   previous state.
//! - **Duplicate detection**: Near-duplicate coordinates are rejected using a scale-aware
//!   Euclidean tolerance based on nearby geometry and floating-point resolution, returning
//!   [`InsertionError::DuplicateCoordinates`](core::algorithms::incremental_insertion::InsertionError::DuplicateCoordinates).
//!   Duplicate UUIDs return
//!   [`InsertionError::DuplicateUuid`](core::algorithms::incremental_insertion::InsertionError::DuplicateUuid).
//! - **Explicit verification**: Use `dt.validate()` for cumulative verification (Levels 1–4), or
//!   `dt.is_valid()` for Level 4 only.

#![expect(
    clippy::multiple_crate_versions,
    reason = "transitive dependency versions are controlled by upstream crates"
)]
// Temporarily allow deprecated warnings during API migrations.
// - Historical Facet -> FacetView and Tds construction migrations
// - DelaunayTriangulation::as_triangulation_mut() removal planned for v0.8.0
// Forbid unsafe code throughout the entire crate
#![forbid(unsafe_code)]

/// Internal low-level triangulation data structures and algorithms.
///
/// This module backs the curated public low-level modules. It includes
/// [`Tds`](crate::core::tds::Tds), [`Cell`](crate::core::cell::Cell),
/// [`FacetView`](crate::core::facet::FacetView),
/// [`Vertex`](crate::core::vertex::Vertex), the generic
/// [`Triangulation`](crate::core::triangulation::Triangulation) wrapper, and
/// algorithm building blocks used by the crate.
///
/// Public docs, examples, benchmarks, and downstream-style tests should prefer
/// the curated public modules and focused preludes:
///
/// - [`crate::tds`] / [`crate::prelude::tds`] for TDS cells, facets, keys,
///   validation reports, and helpers.
/// - [`crate::collections`] / [`crate::prelude::collections`] for public
///   collection aliases and small buffers.
/// - [`crate::algorithms`] / [`crate::prelude::algorithms`] for point-location
///   and conflict-region algorithms.
/// - [`crate::query`] / [`crate::prelude::query`] for read-only traversal,
///   adjacency, convex hull, and set-comparison helpers.
///
/// High-level Delaunay construction and builder APIs live under
/// [`crate::triangulation`] and the focused Delaunay-facing preludes, not under
/// `core`.
#[expect(
    clippy::redundant_pub_crate,
    reason = "`pub(crate)` keeps internal cross-module intent visible while `core` is private"
)]
mod core {
    /// Triangulation algorithms for construction, maintenance, and querying.
    pub mod algorithms {
        /// Flip-based algorithms (Delaunay repair, diagnostics, and related utilities).
        pub mod flips;
        /// Incremental cavity-based insertion.
        pub mod incremental_insertion;
        /// Point location algorithms (facet walking).
        pub mod locate;
        /// Bounded deterministic PL-manifold topology repair.
        pub(crate) mod pl_manifold_repair;
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
    /// - **`SecureHashMap`/`SecureHashSet`**: Use Rust's randomized default
    ///   hasher for collections whose keys are derived from public coordinate
    ///   input or other caller-controlled values.
    ///
    /// ### ⚠️ Security Warning: `DoS` Resistance
    ///
    /// **The hasher used by `FastHashMap`/`FastHashSet` is NOT DoS-resistant.** It should only be
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
    /// Use [`SecureHashMap`](crate::core::collections::SecureHashMap) or
    /// [`SecureHashSet`](crate::core::collections::SecureHashSet) when keys
    /// are derived from public input.
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
    /// use delaunay::prelude::collections::{FastHashMap, FacetToCellsMap, SmallBuffer};
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
    /// ## Key-based internal operations
    ///
    /// The crate uses stable keys (`VertexKey`, `CellKey`) internally for performance.
    /// This module provides optimized maps/sets keyed by those identifiers:
    ///
    /// ```rust
    /// use delaunay::prelude::collections::{CellKeySet, KeyBasedCellMap, VertexKeySet};
    ///
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
            MAX_PRACTICAL_DIMENSION_SIZE, SecureHashMap, SecureHashSet, SmallBuffer, Uuid,
        };

        pub use buffers::*;
        pub use helpers::*;
        pub use key_maps::*;
        pub use secondary_maps::*;
        pub use triangulation_maps::*;
    }
    pub mod edge;
    pub mod facet;
    /// Semantic classification and telemetry for topological operations
    pub mod operations;
    pub mod tds;
    /// Generic triangulation combining kernel + Tds.
    pub mod triangulation;

    /// General utility functions organized by functionality.
    pub mod util {
        pub(crate) mod canonical_points;
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
    }

    // Import concrete internal modules directly via `crate::core::<module>`.
    // Public low-level access is exposed through crate-root facades such as
    // `crate::tds`, `crate::collections`, `crate::algorithms`, and
    // `crate::query`.
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
    #[macro_use]
    pub mod matrix;
    /// Geometric kernel abstraction (CGAL-style).
    pub mod kernel;
    pub mod point;
    pub mod predicates;
    /// Geometric quality measures for d-dimensional simplicial cells
    pub mod quality;
    /// Enhanced predicates with improved numerical robustness
    pub mod robust_predicates;
    /// Simulation of Simplicity (SoS) for deterministic degeneracy resolution
    pub mod sos;
    /// Geometric utility functions for d-dimensional geometry calculations
    pub mod util {
        use crate::geometry::matrix::{MatrixError, StackMatrixDispatchError};
        use crate::geometry::traits::coordinate::CoordinateConversionError;
        use la_stack::LaError;

        // Error types defined here and re-exported from submodules

        /// Errors that can occur during value type conversions.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::geometry::ValueConversionError;
        ///
        /// let err = ValueConversionError::ConversionFailed {
        ///     value: "1.0".to_string(),
        ///     from_type: "f64",
        ///     to_type: "u32",
        ///     details: "out of range".to_string(),
        /// };
        /// assert!(matches!(err, ValueConversionError::ConversionFailed { .. }));
        /// ```
        #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum ValueConversionError {
            /// Failed to convert a value from one type to another
            #[error("Cannot convert {value} from {from_type} to {to_type}: {details}")]
            ConversionFailed {
                /// The value that failed to convert (as string for display)
                value: String,
                /// Source type name
                from_type: &'static str,
                /// Target type name
                to_type: &'static str,
                /// Additional details about the failure
                details: String,
            },
        }

        /// Errors that can occur during random point generation.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::generators::RandomPointGenerationError;
        ///
        /// let err = RandomPointGenerationError::InvalidRange {
        ///     min: "1.0".to_string(),
        ///     max: "0.0".to_string(),
        /// };
        /// assert!(matches!(err, RandomPointGenerationError::InvalidRange { .. }));
        /// ```
        #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum RandomPointGenerationError {
            /// Invalid coordinate range provided
            #[error("Invalid coordinate range: minimum {min} must be less than maximum {max}")]
            InvalidRange {
                /// The minimum value of the range
                min: String,
                /// The maximum value of the range
                max: String,
            },

            /// Failed to generate random value within range
            #[error("Failed to generate random value in range [{min}, {max}]: {details}")]
            RandomGenerationFailed {
                /// The minimum value of the range
                min: String,
                /// The maximum value of the range
                max: String,
                /// Additional details about the failure
                details: String,
            },

            /// Invalid number of points requested
            #[error("Invalid number of points: {n_points} (must be non-negative)")]
            InvalidPointCount {
                /// The invalid number of points requested
                n_points: isize,
            },
        }

        /// Errors that can occur during circumcenter calculation.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::geometry::CircumcenterError;
        ///
        /// let err = CircumcenterError::EmptyPointSet;
        /// assert!(matches!(err, CircumcenterError::EmptyPointSet));
        /// ```
        #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum CircumcenterError {
            /// Empty point set provided
            #[error("Empty point set")]
            EmptyPointSet,

            /// Points do not form a valid simplex
            #[error(
                "Points do not form a valid simplex: expected {expected} points for dimension {dimension}, got {actual}"
            )]
            InvalidSimplex {
                /// Number of points provided
                actual: usize,
                /// Number of points expected (D+1)
                expected: usize,
                /// Dimension
                dimension: usize,
            },

            /// Matrix inversion failed (degenerate simplex)
            #[error("Matrix inversion failed: {details}")]
            MatrixInversionFailed {
                /// Details about the matrix inversion failure
                details: String,
            },

            /// Runtime-dispatched stack matrix dimension is unsupported.
            #[error("Unsupported stack matrix dimension {requested} (maximum supported is {max})")]
            UnsupportedMatrixDimension {
                /// Requested matrix dimension.
                requested: usize,
                /// Maximum supported matrix dimension.
                max: usize,
            },

            /// Internal matrix dispatch requested an active block whose size does not
            /// match the concrete stack matrix.
            ///
            /// Public geometry APIs surface this as a typed error rather than silently
            /// classifying structurally invalid predicate state as degenerate geometry.
            #[error(
                "Active matrix block size {active} does not match concrete matrix dimension {matrix_dimension}"
            )]
            MatrixDimensionMismatch {
                /// Requested active matrix dimension.
                active: usize,
                /// Concrete matrix dimension.
                matrix_dimension: usize,
            },

            /// Linear algebra backend operation failed.
            #[error("Linear algebra failure: {source}")]
            LinearAlgebraFailure {
                /// Typed source error from the linear algebra backend.
                #[from]
                source: LaError,
            },

            /// Matrix operation failed while building or solving a geometry helper matrix.
            #[error("Matrix error: {source}")]
            MatrixError {
                /// Typed source error from matrix operations.
                #[from]
                source: MatrixError,
            },

            /// Array conversion failed
            #[error("Array conversion failed: {details}")]
            ArrayConversionFailed {
                /// Details about the array conversion failure
                details: String,
            },

            /// Coordinate conversion failed while preparing predicate or measure inputs.
            #[error("Coordinate conversion error: {source}")]
            CoordinateConversion {
                /// Typed source error from coordinate conversion.
                #[from]
                source: CoordinateConversionError,
            },

            /// Scalar value conversion failed while converting dimensions or derived measures.
            #[error("Value conversion error: {source}")]
            ValueConversion {
                /// Typed source error from value conversion.
                #[from]
                source: ValueConversionError,
            },
        }

        impl From<StackMatrixDispatchError> for CircumcenterError {
            fn from(source: StackMatrixDispatchError) -> Self {
                match source {
                    StackMatrixDispatchError::UnsupportedDim { k, max } => {
                        Self::UnsupportedMatrixDimension { requested: k, max }
                    }
                    StackMatrixDispatchError::ActiveBlockDimensionMismatch { k, dim } => {
                        Self::MatrixDimensionMismatch {
                            active: k,
                            matrix_dimension: dim,
                        }
                    }
                    StackMatrixDispatchError::La { source } => {
                        Self::LinearAlgebraFailure { source }
                    }
                    StackMatrixDispatchError::Matrix { source } => Self::MatrixError { source },
                }
            }
        }

        /// Error type for surface measure computation operations.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::geometry::{CircumcenterError, SurfaceMeasureError};
        ///
        /// let err = SurfaceMeasureError::GeometryError(CircumcenterError::EmptyPointSet);
        /// assert!(matches!(err, SurfaceMeasureError::GeometryError(_)));
        /// ```
        #[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
        #[non_exhaustive]
        pub enum SurfaceMeasureError {
            /// Error retrieving vertices from a facet.
            #[error("Failed to retrieve facet vertices: {0}")]
            FacetError(#[from] crate::core::facet::FacetError),
            /// Error computing geometry measure.
            #[error("Geometry computation failed: {0}")]
            GeometryError(#[from] CircumcenterError),
        }

        pub mod circumsphere;
        pub mod conversions;
        pub mod measures;
        pub mod norms;
        pub mod point_generation;
        pub mod triangulation_generation;

        // Re-export all public items for ergonomic `crate::geometry::util::*` access.
        pub use circumsphere::*;
        pub use conversions::*;
        pub use measures::*;
        pub use norms::*;
        pub use point_generation::*;
        pub use triangulation_generation::*;
    }
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

pub mod triangulation;

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
/// use delaunay::prelude::triangulation::construction::{
///     DelaunayTriangulation, DelaunayTriangulationConstructionError, vertex,
/// };
/// use delaunay::prelude::topology::validation;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices)?;
///
/// let result = validation::validate_triangulation_euler(dt.tds())?;
/// assert_eq!(result.chi, 1);  // Tetrahedron has χ = 1
/// assert!(result.is_valid());
/// # Ok(())
/// # }
/// ```
pub mod topology {
    /// Traits for topological spaces and error types
    pub mod traits {
        pub(crate) mod global_topology_model;
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
}

/// Public collection aliases and small-buffer types used by low-level APIs.
///
/// This module is the public replacement for reaching through the internal
/// implementation namespace. It keeps common map, set, key-map, and
/// small-buffer aliases convenient without importing every algorithm-specific
/// scratch buffer.
///
/// # Examples
///
/// ```rust
/// use delaunay::collections::{FastHashMap, SmallBuffer};
///
/// let mut counts: FastHashMap<&'static str, usize> = FastHashMap::default();
/// counts.insert("cells", 3);
///
/// let mut scratch: SmallBuffer<usize, 4> = SmallBuffer::new();
/// scratch.push(counts["cells"]);
///
/// assert_eq!(scratch.as_slice(), &[3]);
/// ```
pub mod collections {
    pub use crate::core::collections::{
        CellKeyBuffer, CellKeySet, CellNeighborsMap, CellSecondaryMap, CellToVertexUuidsMap,
        CellVertexBuffer, CellVertexKeysMap, CellVertexUuidBuffer, CellVerticesMap, Entry,
        FacetIndex, FacetIssuesMap, FacetSharingCellsBuffer, FacetToCellsMap, FacetVertexMap,
        FastBuildHasher, FastHashMap, FastHashSet, FastHasher, KeyBasedCellMap, KeyBasedVertexMap,
        MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, PeriodicOffsetBuffer, SecureHashMap,
        SecureHashSet, SimplexVertexBuffer, SmallBuffer, Uuid, UuidToCellKeyMap,
        UuidToVertexKeyMap, VertexKeyBuffer, VertexKeySet, VertexSecondaryMap, VertexToCellsMap,
        VertexUuidBuffer, VertexUuidSet, fast_hash_map_with_capacity, fast_hash_set_with_capacity,
        small_buffer_with_capacity_2, small_buffer_with_capacity_8, small_buffer_with_capacity_16,
    };

    /// Expert aliases for algorithm-local scratch buffers.
    ///
    /// These remain public for advanced users and APIs that expose exact buffer
    /// shapes, but they are separated from the common collection aliases to
    /// avoid accidental broad imports.
    pub mod algorithm_buffers {
        pub use crate::core::collections::{
            BadCellBuffer, CLEANUP_OPERATION_BUFFER_SIZE, CavityBoundaryBuffer, CellRemovalBuffer,
            FacetInfoBuffer, GeometricPointBuffer, PointBuffer, ValidCellsBuffer, ViolationBuffer,
        };
    }
}

/// Public low-level topology data structures and TDS helpers.
///
/// Use this module when you need cells, facets, keys, the
/// [`Tds`](crate::tds::Tds) container, validation reports, or TDS-specific
/// helpers without reaching into the internal implementation namespace.
///
/// # Examples
///
/// ```rust
/// use delaunay::tds::Tds;
///
/// let tds: Tds<f64, (), (), 2> = Tds::empty();
///
/// assert_eq!(tds.number_of_vertices(), 0);
/// assert_eq!(tds.number_of_cells(), 0);
/// ```
pub mod tds {
    pub use crate::core::adjacency::*;
    pub use crate::core::cell::*;
    pub use crate::core::collections::{
        CellKeyBuffer, FacetIndex, FastHashMap, FastHashSet, NeighborBuffer, PeriodicOffsetBuffer,
        SmallBuffer, Uuid,
    };
    pub use crate::core::edge::*;
    pub use crate::core::facet::*;
    pub use crate::core::tds::*;
    pub use crate::core::traits::facet_cache::*;
    pub use crate::core::util::{
        UuidValidationError, checked_facet_key_from_vertex_keys, facet_view_to_vertices,
        facet_views_are_adjacent, format_jaccard_report, jaccard_distance, jaccard_index,
        make_uuid, measure_with_result, stable_hash_u64_slice, usize_to_u8, validate_uuid,
        verify_facet_index_consistency,
    };
    pub use crate::core::vertex::*;
}

/// Public low-level algorithms that are useful outside full construction.
///
/// This module currently exposes point-location and conflict-region building
/// blocks. Higher-level Delaunay construction, repair, and editing APIs remain
/// under [`triangulation`] and the matching focused preludes.
///
/// # Examples
///
/// ```rust
/// use delaunay::algorithms::{LocateError, locate};
/// use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
/// use delaunay::tds::Tds;
///
/// let tds: Tds<f64, (), (), 2> = Tds::empty();
/// let kernel = AdaptiveKernel::new();
/// let point = Point::new([0.0, 0.0]);
///
/// assert!(matches!(
///     locate(&tds, &kernel, &point, None),
///     Err(LocateError::EmptyTriangulation)
/// ));
/// ```
pub mod algorithms {
    pub use crate::core::algorithms::locate::{
        ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
        LocateFallbackReason, LocateResult, LocateStats, extract_cavity_boundary,
        find_conflict_region, locate, locate_with_stats,
    };
}

/// Public read-only traversal, adjacency, convex-hull, and set-comparison APIs.
///
/// This module is intended for callers who need to inspect a triangulation or
/// compare derived topology without importing construction and repair surfaces.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashSet;
///
/// use delaunay::query::{JaccardComputationError, jaccard_index};
///
/// # fn main() -> Result<(), JaccardComputationError> {
/// let a: HashSet<_> = [1, 2, 3].into_iter().collect();
/// let b: HashSet<_> = [3, 4].into_iter().collect();
///
/// let score = jaccard_index(&a, &b)?;
/// assert!((score - 0.25).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
pub mod query {
    pub use crate::assert_jaccard_gte;
    pub use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
    pub use crate::core::algorithms::locate::{
        ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
        LocateFallbackReason, LocateResult, LocateStats, extract_cavity_boundary,
        find_conflict_region, locate, locate_with_stats,
    };
    pub use crate::core::cell::Cell;
    pub use crate::core::edge::EdgeKey;
    pub use crate::core::facet::FacetView;
    pub use crate::core::tds::{CellKey, VertexKey};
    pub use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    pub use crate::core::traits::data_type::{
        DataCopy, DataDebug, DataDeserialize, DataIdentity, DataSerde, DataSerialize, DataType,
    };
    pub use crate::core::triangulation::Triangulation;
    pub use crate::core::util::{
        JaccardComputationError, extract_edge_set, extract_facet_identifier_set,
        extract_hull_facet_set, extract_vertex_coordinate_set, format_jaccard_report,
        jaccard_distance, jaccard_index, measure_with_result,
    };
    pub use crate::core::vertex::Vertex;
    pub use crate::geometry::Point;
    pub use crate::geometry::algorithms::convex_hull::{
        ConvexHull, ConvexHullConstructionError, ConvexHullValidationError,
    };
    pub use crate::geometry::kernel::{
        AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel,
    };
    pub use crate::geometry::traits::coordinate::Coordinate;
    pub use crate::geometry::{insphere, insphere_distance, insphere_lifted};
    pub use crate::triangulation::delaunay::DelaunayTriangulation;
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export from core
    pub use crate::core::{
        adjacency::*,
        cell::*,
        edge::*,
        facet::*,
        tds::*,
        traits::{boundary_analysis::*, data_type::*},
        triangulation::*,
        vertex::*,
    };
    pub use crate::triangulation::delaunay::*;

    // Re-export utility items, but avoid exporting the util module names themselves.
    //
    // In particular, exporting `core::util::uuid` as `uuid` conflicts with the external `uuid`
    // crate name, making `use uuid::Uuid;` ambiguous for downstream users.
    pub use crate::core::util::delaunay_validation::{
        DelaunayValidationError, find_delaunay_violations,
    };
    pub use crate::core::util::{
        deduplication::*, facet_keys::*, facet_utils::*, hashing::*, hilbert::*, jaccard::*,
        measurement::*, uuid::*,
    };

    // Re-export point location algorithms from core::algorithms
    pub use crate::core::algorithms::locate::{
        ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
        LocateFallbackReason, LocateResult, LocateStats, locate, locate_with_stats,
    };

    // Re-export incremental insertion types
    pub use crate::core::algorithms::incremental_insertion::{
        CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind, DelaunayRepairErrorSummary,
        DelaunayRepairFailureContext, HullExtensionReason, InitialSimplexConstructionError,
        InsertionError, InsertionErrorKind, InsertionErrorSourceKind, InsertionErrorSummary,
        NeighborRebuildError, NeighborWiringError, TdsConstructionFailure, TdsValidationFailure,
    };
    pub use crate::core::operations::{InsertionOutcome, InsertionStatistics, SuspicionFlags};

    // Re-export diagnostic types for scientific analysis of construction and repair
    pub use crate::core::algorithms::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairStats,
        DelaunayRepairVerificationContext, FlipContextError, FlipEdgeAdjacencyError, FlipError,
        FlipMutationError, FlipNeighborWiringError, FlipPredicateError, FlipPredicateOperation,
        FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder,
    };

    // Re-export commonly used collection types from core::collections
    // These are frequently used in advanced examples and downstream code
    pub use crate::core::collections::{
        CellNeighborsMap, CellSecondaryMap, FacetToCellsMap, FastHashMap, FastHashSet,
        SecureHashMap, SecureHashSet, SmallBuffer, VertexSecondaryMap, VertexToCellsMap,
        fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    };

    // Re-export from geometry
    pub use crate::geometry::{
        algorithms::*, kernel::*, matrix::*, point::*, predicates::*, quality::*,
        robust_predicates::*, traits::coordinate::*, util::*,
    };

    /// Focused exports for triangulation construction and mutation.
    pub mod triangulation {
        pub use crate::core::operations::{InsertionOutcome, InsertionStatistics, SuspicionFlags};
        pub use crate::core::traits::data_type::{
            DataCopy, DataDebug, DataDeserialize, DataIdentity, DataSerde, DataSerialize, DataType,
        };
        pub use crate::core::triangulation::{
            DuplicateDetectionMetrics, TopologyGuarantee, Triangulation,
            TriangulationConstructionError, TriangulationValidationError, ValidationPolicy,
        };
        pub use crate::core::util::{
            dedup_vertices_epsilon, dedup_vertices_exact, filter_vertices_excluding,
        };
        pub use crate::core::vertex::{
            Vertex, VertexBuilder, VertexBuilderError, VertexValidationError,
        };
        pub use crate::topology::traits::{GlobalTopology, TopologyKind, ToroidalConstructionMode};
        pub use crate::triangulation::builder::*;
        pub use crate::triangulation::delaunay::*;

        /// Batch construction options, builders, and construction errors.
        ///
        /// This focused prelude is for callers configuring Delaunay construction
        /// without importing the broader triangulation editing and repair
        /// surface.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::triangulation::construction::{
        ///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
        /// };
        ///
        /// # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
        /// let vertices = vec![
        ///     vertex!([0.0, 0.0]),
        ///     vertex!([1.0, 0.0]),
        ///     vertex!([0.0, 1.0]),
        /// ];
        /// let triangulation = DelaunayTriangulationBuilder::new(&vertices)
        ///     .build::<()>()?;
        ///
        /// assert_eq!(triangulation.number_of_vertices(), 3);
        /// # Ok(())
        /// # }
        /// ```
        pub mod construction {
            pub use crate::core::triangulation::{
                TopologyGuarantee, Triangulation, TriangulationConstructionError,
            };
            pub use crate::core::vertex::{
                Vertex, VertexBuilder, VertexBuilderError, VertexValidationError,
            };
            pub use crate::topology::traits::{
                GlobalTopology, TopologyKind, ToroidalConstructionMode,
            };
            pub use crate::triangulation::builder::{
                DelaunayTriangulationBuilder, ExplicitConstructionError,
                ExplicitDelaunayValidationError, ExplicitDelaunayValidationErrorKind,
                ExplicitDelaunayValidationSourceKind, ExplicitInsertionError,
                ExplicitInsertionErrorKind, ExplicitInvariantError, ExplicitInvariantErrorKind,
                ExplicitTdsError, ExplicitTdsErrorKind,
            };
            pub use crate::triangulation::delaunay::{
                ConstructionOptions, ConstructionSkipSample, ConstructionSlowInsertionSample,
                ConstructionStatistics, DedupPolicy, DelaunayConstructionFailure,
                DelaunayConstructionRepairPhase, DelaunayRepairPolicy, DelaunayTriangulation,
                DelaunayTriangulationConstructionError,
                DelaunayTriangulationConstructionErrorWithStatistics, InitialSimplexStrategy,
                InsertionOrderStrategy, RetryPolicy,
            };
            // Convenience macro (commonly used in docs/examples).
            pub use crate::vertex;
        }

        /// Bistellar (Pachner) flips for explicit triangulation editing.
        ///
        /// Repair-only diagnostics and validation helpers are intentionally
        /// excluded; use [`crate::prelude::triangulation::repair`] for those.
        ///
        /// ```compile_fail
        /// use delaunay::prelude::triangulation::flips::DelaunayRepairError;
        /// ```
        ///
        pub mod flips {
            pub use crate::core::algorithms::flips::{BistellarMove, ConstK};
            pub use crate::core::collections::{
                CellKeyBuffer, MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer,
            };
            pub use crate::core::tds::{CellKey, VertexKey};
            #[deprecated(
                since = "0.7.7",
                note = "import TopologyGuarantee from delaunay::prelude::triangulation or delaunay::prelude::triangulation::repair"
            )]
            /// Deprecated compatibility re-export; prefer
            /// [`crate::prelude::triangulation::TopologyGuarantee`] or
            /// [`crate::prelude::triangulation::repair::TopologyGuarantee`].
            pub use crate::core::triangulation::TopologyGuarantee;
            pub use crate::core::triangulation::Triangulation;
            pub use crate::triangulation::delaunay::DelaunayTriangulation;
            pub use crate::triangulation::flips::*;

            // Convenience macro (commonly used in docs/examples).
            pub use crate::vertex;
        }

        /// Incremental insertion building blocks and diagnostics.
        ///
        /// Includes compact [`InsertionErrorSummary`] and [`InsertionErrorKind`]
        /// exports for callers that need small by-value diagnostics instead of full insertion
        /// error payloads.
        ///
        /// [`InsertionErrorSummary`]: crate::core::algorithms::incremental_insertion::InsertionErrorSummary
        /// [`InsertionErrorKind`]: crate::core::algorithms::incremental_insertion::InsertionErrorKind
        pub mod insertion {
            pub use crate::core::algorithms::incremental_insertion::{
                CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind,
                DelaunayRepairErrorSummary, DelaunayRepairFailureContext, HullExtensionReason,
                InitialSimplexConstructionError, InsertionError, InsertionErrorKind,
                InsertionErrorSourceKind, InsertionErrorSummary, NeighborRebuildError,
                NeighborWiringError, TdsConstructionFailure, TdsValidationFailure, extend_hull,
                fill_cavity, repair_neighbor_pointers, repair_neighbor_pointers_local,
                wire_cavity_neighbors,
            };
            pub use crate::core::collections::CellKeyBuffer;
            pub use crate::core::facet::FacetHandle;
            pub use crate::core::operations::{
                InsertionOutcome, InsertionResult, InsertionStatistics,
            };
            pub use crate::core::tds::{CellKey, Tds, TdsMutationError, VertexKey};
        }

        /// Topological operation telemetry and repair decisions.
        pub mod operations {
            pub use crate::core::operations::*;
        }

        /// Flip-based Delaunay repair, diagnostics, and Level 4 validation.
        ///
        /// Includes compact [`DelaunayRepairErrorSummary`] and [`DelaunayRepairErrorKind`]
        /// exports for APIs that need repair categories without retaining full repair
        /// diagnostics.
        ///
        /// [`DelaunayRepairErrorSummary`]: crate::core::algorithms::incremental_insertion::DelaunayRepairErrorSummary
        /// [`DelaunayRepairErrorKind`]: crate::core::algorithms::incremental_insertion::DelaunayRepairErrorKind
        pub mod repair {
            pub use crate::core::algorithms::flips::{
                DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairStats,
                DelaunayRepairVerificationContext, FlipContextError, FlipEdgeAdjacencyError,
                FlipError, FlipMutationError, FlipNeighborWiringError, FlipPredicateError,
                FlipPredicateOperation, FlipTriangleAdjacencyError, FlipVertexAdjacencyError,
                RepairQueueOrder, verify_delaunay_for_triangulation,
                verify_delaunay_via_flip_predicates,
            };
            pub use crate::core::algorithms::incremental_insertion::{
                DelaunayRepairErrorKind, DelaunayRepairErrorSummary,
            };
            pub use crate::core::triangulation::{
                TopologyGuarantee, Triangulation, ValidationPolicy,
            };
            pub use crate::core::util::{DelaunayValidationError, find_delaunay_violations};
            pub use crate::triangulation::delaunay::{
                DelaunayCheckPolicy, DelaunayRepairHeuristicConfig, DelaunayRepairHeuristicSeeds,
                DelaunayRepairOperation, DelaunayRepairOutcome, DelaunayRepairPolicy,
                DelaunayTriangulation, DelaunayTriangulationValidationError,
            };
        }

        /// End-to-end "repair then delaunayize" workflow.
        ///
        /// Self-contained: a single `use delaunay::prelude::triangulation::delaunayize::*`
        /// import brings in [`DelaunayTriangulation`], [`vertex!`], and all
        /// delaunayize-specific types.
        pub mod delaunayize {
            pub use crate::core::algorithms::pl_manifold_repair::{
                PlManifoldRepairError, PlManifoldRepairStats,
            };
            pub use crate::triangulation::delaunay::DelaunayTriangulation;
            pub use crate::triangulation::delaunayize::*;

            // Convenience macro (commonly used in docs/examples).
            pub use crate::vertex;
        }

        /// Construction telemetry diagnostics.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::triangulation::diagnostics::ConstructionTelemetry;
        ///
        /// let telemetry = ConstructionTelemetry::default();
        /// assert!(!telemetry.has_data());
        /// ```
        pub mod diagnostics {
            pub use crate::triangulation::diagnostics::{
                BatchLocalRepairTrigger, ConstructionTelemetry, LocalRepairSample,
            };
        }

        /// Validation scheduling helpers for construction diagnostics.
        ///
        /// # Examples
        ///
        /// ```rust
        /// use delaunay::prelude::triangulation::validation::ValidationCadence;
        ///
        /// let cadence = ValidationCadence::from_optional_every(Some(32));
        /// assert!(!cadence.should_validate(31));
        /// assert!(cadence.should_validate(32));
        /// ```
        pub mod validation {
            pub use crate::core::triangulation::{TriangulationValidationError, ValidationPolicy};
            pub use crate::triangulation::delaunay::DelaunayTriangulationValidationError;
            pub use crate::triangulation::validation::*;
        }

        pub use crate::core::algorithms::incremental_insertion::{
            CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind,
            DelaunayRepairErrorSummary, DelaunayRepairFailureContext, HullExtensionReason,
            InsertionError, InsertionErrorKind, InsertionErrorSourceKind, InsertionErrorSummary,
            NeighborWiringError,
        };
        // Convenience macro (commonly used in docs/tests/examples).
        pub use crate::vertex;
    }

    /// Focused exports for collection types used throughout the crate.
    ///
    /// This prelude keeps common map, set, key-map, and small-buffer aliases
    /// convenient without importing every algorithm-specific scratch buffer.
    /// Expert-only buffers remain available from [`crate::collections`]
    /// or the nested [`crate::prelude::collections::algorithm_buffers`] module.
    ///
    /// ```compile_fail
    /// use delaunay::prelude::collections::CellRemovalBuffer;
    /// ```
    pub mod collections {
        pub use crate::core::collections::{
            CellKeyBuffer, CellKeySet, CellNeighborsMap, CellSecondaryMap, CellToVertexUuidsMap,
            CellVertexBuffer, CellVertexKeysMap, CellVertexUuidBuffer, CellVerticesMap, Entry,
            FacetIndex, FacetIssuesMap, FacetSharingCellsBuffer, FacetToCellsMap, FastBuildHasher,
            FastHashMap, FastHashSet, FastHasher, KeyBasedCellMap, KeyBasedVertexMap,
            MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, PeriodicOffsetBuffer, SecureHashMap,
            SecureHashSet, SimplexVertexBuffer, SmallBuffer, Uuid, UuidToCellKeyMap,
            UuidToVertexKeyMap, VertexKeyBuffer, VertexKeySet, VertexSecondaryMap,
            VertexToCellsMap, VertexUuidBuffer, VertexUuidSet, fast_hash_map_with_capacity,
            fast_hash_set_with_capacity, small_buffer_with_capacity_2,
            small_buffer_with_capacity_8, small_buffer_with_capacity_16,
        };

        /// Expert aliases for algorithm-local scratch buffers.
        ///
        /// These remain public for advanced users and for APIs that expose their
        /// exact buffer shapes, but they are separated from the common
        /// collections prelude to avoid accidental broad imports.
        pub mod algorithm_buffers {
            pub use crate::core::collections::{
                BadCellBuffer, CLEANUP_OPERATION_BUFFER_SIZE, CavityBoundaryBuffer,
                CellRemovalBuffer, FacetInfoBuffer, GeometricPointBuffer, PointBuffer,
                ValidCellsBuffer, ViolationBuffer,
            };
        }
    }

    /// Focused exports for low-level topology data structures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds: Tds<f64, (), (), 2> = Tds::empty();
    ///
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// assert_eq!(tds.number_of_cells(), 0);
    /// ```
    pub mod tds {
        pub use crate::core::adjacency::*;
        pub use crate::core::cell::*;
        pub use crate::core::collections::{
            CellKeyBuffer, FacetIndex, FastHashMap, FastHashSet, NeighborBuffer,
            PeriodicOffsetBuffer, SmallBuffer, Uuid,
        };
        pub use crate::core::edge::*;
        pub use crate::core::facet::*;
        pub use crate::core::tds::*;
        pub use crate::core::traits::facet_cache::*;
        pub use crate::core::util::{
            UuidValidationError, checked_facet_key_from_vertex_keys, facet_view_to_vertices,
            facet_views_are_adjacent, format_jaccard_report, jaccard_distance, jaccard_index,
            make_uuid, measure_with_result, stable_hash_u64_slice, usize_to_u8, validate_uuid,
            verify_facet_index_consistency,
        };
        pub use crate::core::vertex::*;
    }

    /// Focused exports for geometry types, predicates, and helpers.
    pub mod geometry {
        pub use crate::geometry::{
            kernel::{AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel},
            matrix::{Matrix, MatrixError, determinant},
            point::Point,
            predicates::{
                InSphere, Orientation, insphere, insphere_distance, insphere_lifted,
                simplex_orientation,
            },
            quality::{
                QualityCellVerticesError, QualityDegeneracyMeasure, QualityError,
                QualityNumericOperation, normalized_volume, radius_ratio,
            },
            robust_predicates::{
                ConsistencyResult, InsphereConsistencyError, robust_insphere, robust_orientation,
            },
            traits::coordinate::{
                Coordinate, CoordinateConversionError, CoordinateIdentity,
                CoordinateRepresentation, CoordinateScalar, CoordinateValidationError,
                DEFAULT_TOLERANCE_F32, DEFAULT_TOLERANCE_F64, DegenerateSimplexReason, FiniteCheck,
                HashCoordinate, OrderedCmp, OrderedEq,
            },
            util::{
                CircumcenterError, SurfaceMeasureError, ValueConversionError, circumcenter,
                circumradius, circumradius_with_center, facet_measure, hypot, inradius,
                safe_coords_from_f64, safe_coords_to_f64, safe_scalar_from_f64, safe_scalar_to_f64,
                safe_usize_to_scalar, simplex_volume, squared_norm, surface_measure,
            },
        };
    }

    /// Focused exports for core algorithms.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::algorithms::{LocateError, locate};
    /// use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
    /// use delaunay::prelude::tds::Tds;
    ///
    /// let tds: Tds<f64, (), (), 2> = Tds::empty();
    /// let kernel = AdaptiveKernel::new();
    /// let point = Point::new([0.0, 0.0]);
    ///
    /// assert!(matches!(
    ///     locate(&tds, &kernel, &point, None),
    ///     Err(LocateError::EmptyTriangulation)
    /// ));
    /// ```
    pub mod algorithms {
        pub use crate::core::algorithms::locate::{
            ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
            LocateFallbackReason, LocateResult, LocateStats, extract_cavity_boundary,
            find_conflict_region, locate, locate_with_stats,
        };
    }

    /// Focused exports for opt-in diagnostic helpers.
    ///
    /// These helpers are compiled only with the `diagnostics` feature because
    /// they are intended for explicit debugging and verification workflows, not
    /// the default public API surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::diagnostics::NeighborSlot;
    ///
    /// assert!(NeighborSlot::Boundary.is_boundary());
    /// ```
    #[cfg(feature = "diagnostics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
    pub mod diagnostics {
        pub use crate::core::algorithms::locate::verify_conflict_region_completeness;
        pub use crate::core::cell::NeighborSlot;
        pub use crate::core::util::{
            DelaunayViolationDetail, DelaunayViolationReport, debug_print_first_delaunay_violation,
            delaunay_violation_report,
        };
    }

    /// Convenience re-exports for common **read-only** workflows (topology traversal, adjacency,
    /// convex-hull extraction, and common input types).
    ///
    /// This is useful if you want a smaller import surface than `delaunay::prelude::*`,
    /// while still having access to the key public APIs typically used in docs/tests/examples/benches.
    ///
    /// Includes:
    /// - Topology traversal: [`DelaunayTriangulation::edges`], [`DelaunayTriangulation::incident_edges`],
    ///   [`DelaunayTriangulation::cell_neighbors`]
    /// - Fast repeated queries: [`DelaunayTriangulation::build_adjacency_index`] and [`AdjacencyIndex`]
    /// - Zero-allocation geometry accessors: [`DelaunayTriangulation::vertex_coords`],
    ///   [`DelaunayTriangulation::cell_vertices`]
    /// - Convex hull extraction: [`ConvexHull::from_triangulation`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashSet;
    ///
    /// use delaunay::prelude::query::{JaccardComputationError, jaccard_index};
    ///
    /// # fn main() -> Result<(), JaccardComputationError> {
    /// let a: HashSet<_> = [1, 2, 3].into_iter().collect();
    /// let b: HashSet<_> = [3, 4].into_iter().collect();
    ///
    /// let score = jaccard_index(&a, &b)?;
    /// assert!((score - 0.25).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    pub mod query {
        // Core read-only traversal / adjacency
        pub use crate::core::adjacency::{AdjacencyIndex, AdjacencyIndexBuildError};
        pub use crate::core::edge::EdgeKey;
        pub use crate::core::tds::{CellKey, VertexKey};
        pub use crate::core::triangulation::Triangulation;
        pub use crate::triangulation::delaunay::DelaunayTriangulation;

        // Locate and conflict-region queries
        pub use crate::core::algorithms::locate::{
            ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
            LocateFallbackReason, LocateResult, LocateStats, extract_cavity_boundary,
            find_conflict_region, locate, locate_with_stats,
        };

        // Common input/output types (kept intentionally small)
        pub use crate::core::cell::Cell;
        pub use crate::core::facet::FacetView;
        pub use crate::core::traits::boundary_analysis::BoundaryAnalysis;
        pub use crate::core::traits::data_type::{
            DataCopy, DataDebug, DataDeserialize, DataIdentity, DataSerde, DataSerialize, DataType,
        };
        pub use crate::core::vertex::Vertex;
        pub use crate::geometry::Point;
        pub use crate::geometry::kernel::{
            AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel,
        };
        pub use crate::geometry::traits::coordinate::Coordinate;

        // Read-only predicates (useful in benchmarks / lightweight geometry checks)
        pub use crate::geometry::{insphere, insphere_distance, insphere_lifted};

        // Read-only algorithms
        pub use crate::assert_jaccard_gte;
        pub use crate::core::util::{
            JaccardComputationError, extract_edge_set, extract_facet_identifier_set,
            extract_hull_facet_set, extract_vertex_coordinate_set, format_jaccard_report,
            jaccard_distance, jaccard_index,
        };
        pub use crate::geometry::algorithms::convex_hull::{
            ConvexHull, ConvexHullConstructionError, ConvexHullValidationError,
        };

        // Instrumentation helpers (no-op unless features enable extra tracking)
        pub use crate::core::util::measure_with_result;

        // Fixture construction macro used by query doctests, examples, and benchmarks.
        pub use crate::vertex;
    }

    /// Focused exports for generating fixture data in doctests, integration tests,
    /// examples, and benchmarks.
    ///
    /// This module is intentionally separate from [`prelude::query`](crate::prelude::query)
    /// so read-only traversal imports do not need to imply random data generation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::generators::{
    ///     RandomPointGenerationError, generate_random_points_seeded,
    /// };
    /// use delaunay::prelude::geometry::Point;
    ///
    /// # fn main() -> Result<(), RandomPointGenerationError> {
    /// let points: Vec<Point<f64, 3>> =
    ///     generate_random_points_seeded(4, (0.0, 1.0), 42)?;
    ///
    /// assert_eq!(points.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub mod generators {
        pub use crate::core::triangulation::TopologyGuarantee;
        pub use crate::geometry::util::{
            RandomPointGenerationError, RandomTriangulationBuilder, generate_grid_points,
            generate_poisson_points, generate_random_points, generate_random_points_in_ball,
            generate_random_points_in_ball_seeded, generate_random_points_periodic,
            generate_random_points_seeded, generate_random_triangulation,
            generate_random_triangulation_with_topology_guarantee, scaled_bounds_by_point_count,
        };
        pub use crate::triangulation::delaunay::InsertionOrderStrategy;
    }

    /// Focused exports for Hilbert ordering and quantization utilities.
    ///
    /// These helpers are useful in doctests, integration tests, examples, and
    /// benchmarks that need deterministic space-filling-curve ordering without
    /// importing the broader triangulation or geometry preludes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::ordering::{hilbert_sorted_indices, HilbertError};
    ///
    /// let coords = [[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
    /// let order = hilbert_sorted_indices(&coords, (0.0, 1.0), 8)?;
    ///
    /// assert_eq!(order.len(), coords.len());
    /// # Ok::<(), HilbertError>(())
    /// ```
    pub mod ordering {
        pub use crate::core::util::{
            HilbertError, hilbert_index, hilbert_indices_prequantized, hilbert_quantize,
            hilbert_sort_by_stable, hilbert_sort_by_unstable, hilbert_sorted_indices,
        };
    }

    /// Topology validation & analysis utilities.
    pub mod topology {
        /// Topology validation utilities.
        pub mod validation {
            pub use crate::topology::TopologyGuarantee;
            pub use crate::topology::characteristics::{euler, validation};
            pub use crate::topology::characteristics::{euler::*, validation::*};
            pub use crate::topology::manifold::{
                ManifoldError, validate_closed_boundary, validate_facet_degree,
                validate_ridge_links, validate_ridge_links_for_cells, validate_vertex_links,
            };
            pub use crate::topology::traits::*;
        }

        /// Topological space models and traits.
        pub mod spaces {
            pub use crate::topology::spaces::*;
            pub use crate::topology::traits::{
                GlobalTopology, TopologicalSpace, TopologyError, TopologyKind,
                ToroidalConstructionMode,
            };
        }
    }

    // Convenience macros
    pub use crate::vertex;
}

/// The function `is_normal` checks that structs implement `auto` traits.
/// Traits are checked at compile time, so this function is only used for
/// testing.
#[must_use]
pub const fn is_normal<T: Send + Sync + Unpin>() -> bool {
    true
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::{
        core::{
            adjacency::AdjacencyIndex, cell::Cell, edge::EdgeKey, tds::Tds,
            triangulation::Triangulation, vertex::Vertex,
        },
        geometry::{
            Point, algorithms::convex_hull::ConvexHull, kernel::AdaptiveKernel, kernel::FastKernel,
        },
        is_normal,
        prelude::triangulation::delaunayize::{
            CellValidationError, DelaunayTriangulationConstructionError, DelaunayizeConfig,
            DelaunayizeError, DelaunayizeOutcome, PlManifoldRepairError, PlManifoldRepairStats,
        },
        prelude::triangulation::repair::{
            DelaunayCheckPolicy, DelaunayRepairError, DelaunayRepairOutcome, DelaunayRepairPolicy,
            DelaunayRepairStats, DelaunayTriangulation as RepairDelaunayTriangulation,
            FlipContextError, FlipError, RepairQueueOrder, TopologyGuarantee,
            verify_delaunay_for_triangulation, verify_delaunay_via_flip_predicates,
        },
        prelude::*,
        triangulation::delaunay::DelaunayTriangulation,
        vertex,
    };

    #[cfg(feature = "count-allocations")]
    use allocation_counter::measure;

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
        assert!(is_normal::<DelaunayizeConfig>());
        assert!(is_normal::<DelaunayizeOutcome<f64, (), (), 3>>());
        assert!(is_normal::<DelaunayizeError>());
        assert!(is_normal::<DelaunayRepairError>());
        assert!(is_normal::<DelaunayRepairStats>());
        assert!(is_normal::<PlManifoldRepairError>());
        assert!(is_normal::<PlManifoldRepairStats<f64, (), (), 3>>());
        assert!(is_normal::<CellValidationError>());
        assert!(is_normal::<DelaunayTriangulationConstructionError>());
    }

    #[test]
    fn prelude_collections_exports() {
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
    fn prelude_repair_exports() {
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let dt: RepairDelaunayTriangulation<_, (), (), 2> =
            RepairDelaunayTriangulation::new(&vertices).unwrap();
        let kernel = AdaptiveKernel::<f64>::new();

        assert!(verify_delaunay_for_triangulation(dt.as_triangulation()).is_ok());
        assert!(verify_delaunay_via_flip_predicates(dt.tds(), &kernel).is_ok());

        let stats = DelaunayRepairStats::default();
        let outcome = DelaunayRepairOutcome {
            stats: stats.clone(),
            heuristic: None,
        };
        assert_eq!(outcome.stats.flips_performed, stats.flips_performed);
        let order = RepairQueueOrder::Fifo;
        assert!(matches!(order, RepairQueueOrder::Fifo));
        assert_eq!(
            DelaunayRepairPolicy::default(),
            DelaunayRepairPolicy::EveryInsertion
        );
        assert_eq!(DelaunayCheckPolicy::default(), DelaunayCheckPolicy::EndOnly);

        let err = DelaunayRepairError::Flip(FlipError::DegenerateCell);
        assert!(matches!(err, DelaunayRepairError::Flip(_)));
        let context_err = FlipContextError::ReplacementPeriodicOffsetCountMismatch {
            cell_count: 1,
            offset_count: 0,
        };
        assert!(matches!(
            context_err,
            FlipContextError::ReplacementPeriodicOffsetCountMismatch { .. }
        ));
        let topo = TopologyGuarantee::PLManifold;
        assert!(matches!(topo, TopologyGuarantee::PLManifold));
    }

    #[test]
    fn prelude_quality_exports() {
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
            assert!(tds.cell(cell_key).is_some());
        }
    }

    #[test]
    fn test_prelude_point_location() {
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
    fn basic_alloc_counting() {
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
    fn alloc_counting_with_vec() {
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
