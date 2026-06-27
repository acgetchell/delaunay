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
//!   - The validation hierarchy and invariant stack (Levels 1–5)
//!   - Topological guarantees (`TopologyGuarantee`) and insertion-time validation policy (`ValidationPolicy`)
//!   - High-level error semantics and programming contract (transactional operations, duplicate rejection)
//!
//! - **docs/workflows.md**:
//!   Task-oriented, end-to-end usage recipes (Builder API, Edit API, validation,
//!   repairs, diagnostics, and statistics).
//!
//! - **docs/validation.md**:
//!   Formal definitions of validation Levels 1–5, their costs, and guidance on when
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
//! | Construct/configure a Delaunay triangulation | `use delaunay::prelude::construction::*` |
//! | Build/validate/repair generic triangulations | `use delaunay::prelude::triangulation::*` |
//! | Low-level incremental insertion building blocks | `use delaunay::prelude::insertion::*` |
//! | Post-construction vertex deletion errors and keys | `use delaunay::prelude::deletion::*` |
//! | Read-only queries, traversal, convex hull | `use delaunay::prelude::query::*` |
//! | Point location and conflict-region algorithms | `use delaunay::prelude::algorithms::*` |
//! | Geometry helpers, coordinate ranges, predicates, points | `use delaunay::prelude::geometry::*` |
//! | Random points / triangulations for examples and tests | `use delaunay::prelude::generators::*` |
//! | Hilbert ordering and quantization utilities | `use delaunay::prelude::ordering::*` |
//! | Unified Pachner move workflow | `use delaunay::prelude::pachner::*` |
//! | Delaunay repair and flip-based Level 5 validation | `use delaunay::prelude::repair::*` |
//! | Delaunayize workflow (repair + flip) | `use delaunay::prelude::delaunayize::*` |
//! | Construction telemetry diagnostics | `use delaunay::prelude::diagnostics::*` |
//! | Validation policies, errors, reports, and Level 5 diagnostics | `use delaunay::prelude::validation::*` |
//! | Topology validation, Euler characteristic, ridge queries | `use delaunay::prelude::topology::validation::*` |
//! | Topological spaces, topology traits, lifted toroidal IDs | `use delaunay::prelude::topology::spaces::*` |
//! | Low-level TDS simplices, facets, keys | `use delaunay::prelude::tds::*` |
//! | Collection types (`FastHashMap`, etc.) | `use delaunay::prelude::collections::*` |
//! | Broad convenience import for exploratory code | `use delaunay::prelude::*` |
//!
//! ## Public low-level namespace policy
//!
//! High-level Delaunay APIs are available directly from the crate root and
//! focused root modules: [`DelaunayTriangulation`], [`DelaunayTriangulationBuilder`],
//! [`construction`](crate::construction), [`flips`](crate::flips),
//! [`repair`](crate::repair), [`validation`](crate::validation), and
//! [`delaunayize`](crate::delaunayize).  The nested `delaunay::delaunay::*`
//! facade is intentionally not part of the public API; use the crate root or a
//! focused prelude instead.
//!
//! ```compile_fail
//! use delaunay::delaunay::DelaunayTriangulation;
//! ```
//!
//! The low-level implementation namespace is private. The public low-level
//! surface is exposed through curated modules:
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
//! ### Validation hierarchy (Levels 1–5)
//!
//! ```rust
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, vertex,
//! };
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0, 0.0]?,
//!     vertex![1.0, 0.0, 0.0]?,
//!     vertex![0.0, 1.0, 0.0]?,
//!     vertex![0.0, 0.0, 1.0]?,
//! ];
//! let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // Levels 1–2: elements + structural (TDS)
//! assert!(dt.tds().validate().is_ok());
//!
//! // Levels 1–3: elements + structural + topology
//! assert!(dt.as_triangulation().validate().is_ok());
//!
//! // Levels 1–4: elements + structural + topology + faithful embedding
//! assert!(dt.as_triangulation().validate_embedding().is_ok());
//!
//! // Level 5 only: Delaunay property (assumes Levels 1–4)
//! assert!(dt.is_valid_delaunay().is_ok());
//!
//! // Levels 1–5: full cumulative validation
//! assert!(dt.validate().is_ok());
//! # Ok(())
//! # }
//! ```
//!
//! ### Topology guarantees and insertion-time validation (`TopologyGuarantee`, `ValidationPolicy`)
//!
//! ```rust
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee, vertex,
//! };
//! use delaunay::prelude::validation::ValidationPolicy;
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0, 0.0]?,
//!     vertex![1.0, 0.0, 0.0]?,
//!     vertex![0.0, 1.0, 0.0]?,
//!     vertex![0.0, 0.0, 1.0]?,
//! ];
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! assert_eq!(dt.topology_guarantee(), TopologyGuarantee::PLManifold);
//! assert_eq!(dt.validation_policy(), ValidationPolicy::ExplicitOnly);
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
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, vertex,
//! };
//! use delaunay::prelude::insertion::InsertionError;
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0]?,
//!     vertex![1.0, 0.0]?,
//!     vertex![0.0, 1.0]?,
//! ];
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! let before_vertices = dt.number_of_vertices();
//! let before_simplices = dt.number_of_simplices();
//!
//! // Duplicate coordinates are rejected.
//! let result = dt.insert_vertex(vertex![0.0, 0.0]?);
//! std::assert_matches!(result, Err(InsertionError::DuplicateCoordinates { .. }));
//!
//! // On error, the triangulation is unchanged.
//! assert_eq!(dt.number_of_vertices(), before_vertices);
//! assert_eq!(dt.number_of_simplices(), before_simplices);
//! # Ok(())
//! # }
//! ```
//!
//! # Triangulation invariants and validation hierarchy
//!
//! The crate is organized as a small **validation stack**, where each layer adds additional
//! invariants on top of the preceding one:
//!
//! - [`Vertex`](crate::tds::Vertex) and [`Simplex`](crate::tds::Simplex) provide
//!   **element validity** checks.
//!   Level 1 (elements) validation checks invariants such as:
//!   - **Vertex coordinates** – finite (no NaN/∞) and UUID is non-nil.
//!   - **Simplex shape** – exactly D+1 distinct vertex keys, valid UUID, and neighbor buffer length
//!     (if present) is D+1.
//!
//!   These checks are surfaced via [`Vertex::is_valid`](crate::tds::Vertex::is_valid),
//!   [`Vertex::vertex_report`](crate::tds::Vertex::vertex_report),
//!   [`Simplex::is_valid`](crate::tds::Simplex::is_valid), and
//!   [`Simplex::simplex_report`](crate::tds::Simplex::simplex_report), and are automatically run by
//!   [`Tds::validate`](crate::tds::Tds::validate) (Levels 1–2).
//!
//! - [`Tds`](crate::tds::Tds) (Triangulation Data Structure)
//!   stores the **combinatorial / structural** representation.
//!   Level 2 (structural) validation checks invariants such as:
//!   - **Vertex mappings** – every vertex UUID has a corresponding key and vice versa.
//!   - **Simplex mappings** – every simplex UUID has a corresponding key and vice versa.
//!   - **No duplicate simplices** – no two maximal simplices share the same vertex set.
//!   - **Facet incidence** – each facet is one-sided or two-sided; topology
//!     metadata decides whether a one-sided facet is semantic boundary.
//!   - **Neighbor consistency** – neighbor relationships are mutual and reference a shared facet.
//!
//!   These checks are surfaced via [`Tds::is_valid`](crate::tds::Tds::is_valid)
//!   (structural only) and [`Tds::validate`](crate::tds::Tds::validate)
//!   (Levels 1–2, elements + structural). For cumulative diagnostics across the full stack,
//!   use [`DelaunayTriangulation::validation_report`](crate::DelaunayTriangulation::validation_report).
//!
//! - [`Triangulation`] builds on the TDS and validates
//!   **manifold topology**.
//!   Level 3 (topology) validation is performed by
//!   [`Triangulation::is_valid_topology`](crate::Triangulation::is_valid_topology) (Level 3 only) and
//!   [`Triangulation::validate`](crate::Triangulation::validate) (Levels 1–3), which:
//!   - Strengthens facet incidence to the **manifold facet property**:
//!     one-sided facets are valid only when the declared topology admits
//!     boundary; two-sided facets are interior.
//!   - Checks the **Euler characteristic** of the triangulation (using the topology module).
//!
//! - [`Triangulation`] also validates the **faithfulness of the embedding** in
//!   the active affine chart. Level 4 (embedding) validation is performed by
//!   [`Triangulation::is_valid_embedding`](crate::Triangulation::is_valid_embedding) (Level 4 only) and
//!   [`Triangulation::validate_embedding`](crate::Triangulation::validate_embedding) (Levels 1–4).
//!   Euclidean topology is checked directly in its ambient chart; toroidal
//!   topology is checked in periodic covering-space charts.
//!
//! - [`DelaunayTriangulation`] builds on
//!   `Triangulation` and validates the **geometric** Delaunay condition.
//!   Level 5 (Delaunay property) validation is performed by
//!   [`DelaunayTriangulation::is_valid_delaunay`](crate::DelaunayTriangulation::is_valid_delaunay) (Level 5 only) and
//!   [`DelaunayTriangulation::validate`](crate::DelaunayTriangulation::validate) (Levels 1–5).
//!   Batch construction runs final Delaunay validation before returning.
//!   Incremental insertion can run global Level 5 checks according to
//!   [`DelaunayCheckPolicy`](crate::repair::DelaunayCheckPolicy). If robust
//!   fallback and repair cannot certify a checked result, the operation returns a
//!   typed error rather than silently accepting a known violation.
//!
//! ## Validation
//!
//! The crate exposes five validation levels
//! (element → structural → topology → embedding → Delaunay). The
//! canonical guide (when to use each level, complexity, examples, troubleshooting) lives in
//! `docs/validation.md`:
//! <https://github.com/acgetchell/delaunay/blob/main/docs/validation.md>
//!
//! In brief:
//! - Level 1 (elements / `Vertex` + `Simplex`): `Vertex::is_valid()` /
//!   `Simplex::is_valid()` for fast checks, or `vertex_report()` /
//!   `simplex_report()` for element-local diagnostics.
//! - Level 2 (structural / `Tds`): `dt.tds().is_valid()` for a quick check, or `dt.tds().validate()` for
//!   Levels 1–2.
//! - Level 3 (topology / `Triangulation`): `dt.as_triangulation().is_valid_topology()` for topology-only checks, or
//!   `dt.as_triangulation().validate()` for Levels 1–3.
//! - Level 4 (embedding / `Triangulation`): `dt.as_triangulation().validate_embedding()` for cumulative
//!   faithful embedded-geometry checks, or `dt.as_triangulation().embedding_report()` for layer-local diagnostics.
//! - Level 5 (Delaunay / `DelaunayTriangulation`): `dt.is_valid_delaunay()` for the Delaunay
//!   property, or `dt.delaunay_report()` for layer-local diagnostics.
//! - Cumulative Delaunay validation: `dt.validate()` for Levels 1–5, or
//!   `dt.validation_report()` for full diagnostics.
//!
//! ### Automatic topology validation during insertion (`ValidationPolicy`)
//!
//! In addition to explicit validation calls, incremental construction (`new()` / `insert*()`) can run an
//! automatic **Level 3** topology validation pass after insertion, controlled by
//! [`ValidationPolicy`](crate::prelude::validation::ValidationPolicy).
//!
//! The initial policy is derived from the active topology guarantee. The default
//! [`TopologyGuarantee::PLManifold`](crate::prelude::TopologyGuarantee::PLManifold)
//! uses [`ValidationPolicy::ExplicitOnly`](crate::prelude::validation::ValidationPolicy::ExplicitOnly):
//! mandatory local topology checks still run during insertion, while full Level 3 validation is a
//! caller-owned explicit checkpoint.
//!
//! This automatic pass only runs Level 3 (`Triangulation::is_valid_topology()`). It does **not** run
//! Level 4 embedding validation or Level 5 Delaunay validation.
//!
//! ```rust
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, vertex,
//! };
//! use delaunay::prelude::validation::ValidationPolicy;
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0, 0.0]?,
//!     vertex![1.0, 0.0, 0.0]?,
//!     vertex![0.0, 1.0, 0.0]?,
//!     vertex![0.0, 0.0, 1.0]?,
//! ];
//! let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
//!
//! // Caller-owned validation mode: keep mandatory topology checks, but run full
//! // Level 3 validation only through explicit validation calls.
//! dt.try_set_validation_policy(ValidationPolicy::ExplicitOnly)?;
//!
//! // Do incremental work...
//! dt.insert_vertex(vertex![0.2, 0.2, 0.2]?)?;
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
//! [`TopologyGuarantee`](crate::prelude::construction::TopologyGuarantee). This is separate from
//! `ValidationPolicy`: it controls *what* invariants Level 3 enforces, not *when* automatic
//! validation runs.
//!
//! - [`TopologyGuarantee::PLManifold`](crate::prelude::construction::TopologyGuarantee::PLManifold)
//!   (default): enforces manifold facet degree, boundary closure, connectedness, Euler characteristic,
//!   and link-based manifold conditions. Ridge-link checks are applied incrementally during insertion,
//!   with vertex-link validation performed at construction completion.
//!
//!   The formal topological definitions, link conditions, and rationale for this validation strategy
//!   are documented in `docs/invariants.md`.
//! - [`TopologyGuarantee::PLManifoldStrict`](crate::prelude::construction::TopologyGuarantee::PLManifoldStrict):
//!   vertex-link validation after every insertion (slowest, maximum safety).
//! - [`TopologyGuarantee::Pseudomanifold`](crate::prelude::construction::TopologyGuarantee::Pseudomanifold):
//!   skips vertex-link validation (may be faster), but bistellar flip convergence is not guaranteed and
//!   you may want to validate the Delaunay property explicitly for near-degenerate inputs.
//!
//! ```rust
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, vertex,
//! };
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0, 0.0]?,
//!     vertex![1.0, 0.0, 0.0]?,
//!     vertex![0.0, 1.0, 0.0]?,
//!     vertex![0.0, 0.0, 1.0]?,
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
//! use delaunay::prelude::construction::{
//!     DelaunayResult, DelaunayTriangulationBuilder, vertex,
//! };
//!
//! # fn main() -> DelaunayResult<()> {
//! let vertices = vec![
//!     vertex![0.0, 0.0, 0.0]?,
//!     vertex![1.0, 0.0, 0.0]?,
//!     vertex![0.0, 1.0, 0.0]?,
//!     vertex![0.0, 0.0, 1.0]?,
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
//! ### Coordinate scalar policy
//!
//! The default supported coordinate input type is `f64`, matching the crate's
//! current linear algebra backend and geometric-primitive correctness
//! guarantees. Exact arithmetic is already used internally for robust predicate
//! fallbacks, and exact coordinate input may be supported explicitly in the
//! future.
//!
//! # Programming contract (high-level)
//!
//! - **Transactional mutations**: Construction and incremental operations are designed to be
//!   all-or-nothing. If an operation returns `Err(_)`, the triangulation is rolled back to its
//!   previous state.
//! - **Duplicate detection**: Near-duplicate coordinates are rejected using a scale-aware
//!   Euclidean tolerance based on nearby geometry and floating-point resolution, returning
//!   [`InsertionError::DuplicateCoordinates`](crate::prelude::insertion::InsertionError::DuplicateCoordinates).
//!   Duplicate UUIDs return
//!   [`InsertionError::DuplicateUuid`](crate::prelude::insertion::InsertionError::DuplicateUuid).
//! - **Explicit verification**: Use `dt.validate()` for cumulative verification (Levels 1–5), or
//!   `dt.is_valid_delaunay()` for Level 5 only.

#![expect(
    clippy::multiple_crate_versions,
    reason = "transitive dependency versions are controlled by upstream crates"
)]
// Forbid unsafe code throughout the entire crate
#![forbid(unsafe_code)]

/// Internal low-level triangulation data structures and algorithms.
///
/// This module backs the curated public low-level modules. It includes
/// [`Tds`](crate::tds::Tds), [`Simplex`](crate::tds::Simplex),
/// [`FacetView`](crate::tds::FacetView),
/// [`Vertex`](crate::tds::Vertex), the generic
/// [`Triangulation`] wrapper, and
/// algorithm building blocks used by the crate.
///
/// Public docs, examples, benchmarks, and downstream-style tests should prefer
/// the curated public modules and focused preludes:
///
/// - [`crate::tds`] / [`crate::prelude::tds`] for TDS simplices, facets, keys,
///   validation reports, and helpers.
/// - [`crate::collections`] / [`crate::prelude::collections`] for public
///   collection aliases and small buffers.
/// - [`crate::algorithms`] / [`crate::prelude::algorithms`] for point-location
///   and conflict-region algorithms.
/// - [`crate::query`] / [`crate::prelude::query`] for read-only traversal,
///   adjacency, convex hull, and set-comparison helpers.
///
/// High-level Delaunay construction and builder APIs live at the crate root
/// and under the focused Delaunay-facing preludes, not under `core`.
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
    pub mod facet_incidence;
    pub mod simplex;
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
    /// **Misuse patterns:**
    /// - Processing untrusted coordinate data from external sources
    /// - Using user-provided keys without validation
    /// - Network-facing applications with external input
    ///
    /// Use [`SecureHashMap`](crate::collections::SecureHashMap) or
    /// [`SecureHashSet`](crate::collections::SecureHashSet) when keys
    /// are derived from public input.
    ///
    /// ## Small Collections
    ///
    /// - **`SmallVec`**: Uses stack allocation for small collections, avoiding heap
    ///   allocations for the common case where collections remain small. This is
    ///   particularly effective for:
    ///   - Vertex neighbor lists (typically D+1 neighbors)
    ///   - Facet-to-simplex mappings (typically 1-2 simplices per facet)
    ///   - Temporary collections during geometric operations
    ///
    /// # Usage Patterns
    ///
    /// The size parameters for `SmallVec` are chosen based on empirical analysis of
    /// typical triangulation patterns:
    ///
    /// - **2 elements**: Facet incidence (one-sided = 1 simplex, two-sided = 2 simplices)
    /// - **4 elements**: Small temporary collections during geometric operations
    /// - **8 elements**: Vertex degrees and simplex neighbor counts in typical triangulations
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
    /// use delaunay::prelude::collections::{FastHashMap, SmallBuffer};
    ///
    /// // Use optimized HashMap for temporary mappings
    /// let mut temp_map: FastHashMap<u64, usize> = FastHashMap::default();
    /// temp_map.insert(7, 3);
    ///
    /// // Use stack-allocated buffer for small collections
    /// let mut small_list: SmallBuffer<i32, 8> = SmallBuffer::new();
    /// small_list.push(1);
    /// small_list.push(2);
    ///
    /// assert_eq!(temp_map.len(), 1);
    /// ```
    ///
    /// ## Key-based internal operations
    ///
    /// The crate uses stable keys (`VertexKey`, `SimplexKey`) internally for performance.
    /// This module provides optimized maps/sets keyed by those identifiers:
    ///
    /// ```rust
    /// use delaunay::prelude::collections::{SimplexKeySet, KeyBasedSimplexMap, VertexKeySet};
    ///
    /// let mut internal_simplices: SimplexKeySet = SimplexKeySet::default();
    /// let mut internal_vertices: VertexKeySet = VertexKeySet::default();
    /// let mut key_mappings: KeyBasedSimplexMap<String> = KeyBasedSimplexMap::default();
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
    /// Generic triangulation construction helpers.
    pub mod construction;
    pub mod edge;
    /// Embedded Euclidean geometry validation for generic triangulations.
    pub mod embedding;
    pub mod facet;
    /// Incremental insertion for generic triangulations.
    pub mod insertion;
    /// Semantic classification and telemetry for topological operations
    pub mod operations;
    /// Geometric orientation validation and canonicalization for generic triangulations.
    pub mod orientation;
    /// Read-only query and traversal helpers for generic triangulations.
    pub mod query;
    /// Local topology repair for generic triangulations.
    pub mod repair;
    /// Scoped rollback guards for internal topology mutation windows.
    pub(crate) mod rollback;
    /// Triangulation data structure internals.
    pub mod tds {
        mod equality;
        pub mod errors;
        pub(crate) mod incidence;
        mod keys;
        mod mutation;
        pub(crate) mod rollback;
        mod snapshot;
        mod storage;
        mod validation;

        pub use errors::*;
        pub use keys::{SimplexKey, VertexKey};
        pub(crate) use rollback::{
            TdsOwnerRollbackTransaction, TdsRollbackOwner, TdsRollbackTransaction,
        };
        pub use storage::Tds;
    }
    /// Generic triangulation combining kernel + Tds.
    pub mod triangulation;
    /// Generic validation orchestration for triangulations.
    pub mod validation;

    /// General utility functions organized by functionality.
    pub mod util {
        pub(crate) mod canonical_points;
        pub mod deduplication;
        pub mod facet_keys;
        pub mod facet_utils;
        pub mod hashing;
        pub mod hilbert;
        pub mod jaccard;
        pub mod measurement;
        pub mod uuid;

        // Re-export utility internals within the private core namespace.
        pub use deduplication::*;
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
        pub mod data_type;
        pub mod facet_cache;
        pub mod facet_incidence_analysis;
        pub use data_type::*;
    }

    // Import concrete internal modules directly via `crate::core::<module>`.
    // Public low-level access is exposed through crate-root facades such as
    // `crate::tds`, `crate::collections`, `crate::algorithms`, and
    // `crate::query`.
}

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench_fixtures;

/// Contains geometric types including the `Point` struct and geometry predicates.
///
/// The geometry module provides coordinate abstractions through the
/// [`Coordinate`](crate::geometry::traits::coordinate::Coordinate) trait,
/// [`CoordinateRange`](crate::geometry::coordinate_range::CoordinateRange)
/// value type, and [`Point`](crate::geometry::point::Point) type. The default
/// supported coordinate input type is `f64`, matching the crate's current
/// linear algebra backend and geometric-primitive correctness guarantees;
/// exact coordinate input may be supported explicitly in the future.
pub mod geometry {
    /// Geometric algorithms for triangulations and spatial data structures
    pub mod algorithms {
        /// Convex hull operations on d-dimensional triangulations
        pub mod convex_hull;
        pub use convex_hull::*;
    }
    /// Validated coordinate-range types.
    pub mod coordinate_range;
    // Pure Level 4 embedding predicates are crate-internal implementation
    // machinery; downstream users should use `Triangulation::embedding_report`.
    pub(crate) mod embedding;
    #[macro_use]
    pub mod matrix;
    /// Geometric kernel abstraction (CGAL-style).
    pub mod kernel;
    pub mod point;
    pub mod predicates;
    /// Geometric quality measures for d-dimensional simplices
    pub mod quality;
    /// Enhanced predicates with improved numerical robustness
    pub mod robust_predicates;
    /// Simulation of Simplicity (SoS) for deterministic degeneracy resolution
    pub mod sos;
    /// Geometric utility functions for d-dimensional geometry calculations
    pub mod util {
        pub mod circumsphere;
        pub mod conversions;
        pub mod measures;
        pub mod norms;
        pub mod point_generation;
        pub mod triangulation_generation;

        // Re-export all public utility items for ergonomic `crate::geometry::util::*` access.
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
    pub use coordinate_range::*;
    pub use matrix::*;
    pub use point::*;
    pub use predicates::*;
    pub use quality::*;
    pub use traits::*;
    pub use util::*;
}

/// Fluent builder for Delaunay triangulations.
#[path = "delaunay/builder.rs"]
pub mod builder;
/// Batch construction options, errors, statistics, and policy helpers.
#[path = "delaunay/construction.rs"]
pub mod construction;
/// TDS-level implementation helpers for Level 5 Delaunay-property scans.
#[path = "delaunay/property_validation.rs"]
mod delaunay_property_validation;
/// Read-only Delaunay query, traversal, and accessor methods.
#[path = "delaunay/query.rs"]
pub(crate) mod delaunay_query;
/// Delaunay-level rollback guards for mutation windows with auxiliary state.
#[path = "delaunay/rollback.rs"]
pub(crate) mod delaunay_rollback;
/// End-to-end "repair then delaunayize" workflow.
#[path = "delaunay/delaunayize.rs"]
pub mod delaunayize;
/// Post-construction vertex deletion operations.
#[path = "delaunay/deletion.rs"]
pub(crate) mod deletion;
/// Construction and performance diagnostics.
#[path = "delaunay/diagnostics.rs"]
pub mod diagnostics;
/// Triangulation editing operations (bistellar flips).
#[path = "delaunay/flips.rs"]
pub mod flips;
/// Post-construction vertex insertion operations.
#[path = "delaunay/insertion.rs"]
pub(crate) mod insertion;
#[path = "delaunay/locality.rs"]
pub(crate) mod locality;
/// Unified Pachner move workflow API for Monte-Carlo-style editing.
#[path = "delaunay/pachner.rs"]
pub mod pachner;
/// Repair policies and outcomes for Delaunay triangulations.
#[path = "delaunay/repair.rs"]
pub mod repair;
/// Serialization support for Delaunay triangulations.
#[path = "delaunay/serialization.rs"]
pub(crate) mod serialization;
/// Delaunay triangulation layer with incremental insertion.
#[path = "delaunay/triangulation.rs"]
pub(crate) mod triangulation;
/// Delaunay-level validation APIs, reports, and construction diagnostics.
#[path = "delaunay/validation.rs"]
pub mod validation;

/// I/O and downstream-facing export data models.
pub mod io {
    /// Generic simplicial-complex export data for notebooks and downstream tools.
    pub mod visualization;

    pub use visualization::*;
}

// Re-export commonly used Delaunay-facing types at the crate root.
pub use crate::builder::DelaunayTriangulationBuilder;
pub use crate::construction::{
    ConstructionOptions, ConstructionSkipSample, ConstructionSlowInsertionSample,
    ConstructionStatistics, DedupPolicy, DedupTolerance, DelaunayConstructionFailure,
    DelaunayConstructionRepairPhase, DelaunayConstructionRetryFailure, DelaunayError,
    DelaunayResult, DelaunayTriangulationConstructionError,
    DelaunayTriangulationConstructionErrorWithStatistics, InitialSimplexStrategy,
    InsertionOrderStrategy, RetryPolicy,
};
pub use crate::core::algorithms::incremental_insertion::{
    CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind, DelaunayRepairFailureContext,
    HullExtensionReason, InitialSimplexConstructionError, InitialSimplexUnexpectedInsertionStage,
    InsertionError, InsertionErrorKind, InsertionErrorSourceKind,
    InsertionTopologyValidationContext, NeighborRebuildError, NeighborWiringError,
    SpatialIndexConstructionFailure, TdsConstructionFailure, TdsValidationFailure, extend_hull,
    fill_cavity, repair_neighbor_pointers, repair_neighbor_pointers_local, wire_cavity_neighbors,
};
pub use crate::core::algorithms::pl_manifold_repair::{
    PlManifoldRepairError, PlManifoldRepairStats,
};
pub use crate::core::construction::{
    FinalDelaunayValidationContext, FinalTopologyValidationContext, TriangulationConstructionError,
};
pub use crate::core::embedding::{
    PeriodicDomainPeriodError, TriangulationEmbeddingIntersectionDetail,
    TriangulationEmbeddingSimplexDetail, TriangulationEmbeddingSimplexPairDetail,
    TriangulationEmbeddingValidationError, TriangulationEmbeddingValidationErrorKind,
    TriangulationEmbeddingValidationReport,
};
pub use crate::core::insertion::DuplicateDetectionMetrics;
pub use crate::core::operations::{
    InsertionOutcome, InsertionResult, InsertionStatistics, RepairDecision, RepairSkipReason,
    SuspicionFlags, TopologicalOperation,
};
pub use crate::core::triangulation::Triangulation;
pub use crate::core::util::DeduplicationError;
pub use crate::core::validation::{
    TopologyGuarantee, TriangulationValidationError, ValidationConfigurationError, ValidationPolicy,
};
#[cfg(feature = "diagnostics")]
pub use crate::delaunay_property_validation::debug_print_first_delaunay_violation;
pub use crate::delaunay_property_validation::{
    DelaunayValidationError, DelaunayViolationDetail, DelaunayViolationReport,
    delaunay_violation_report, find_delaunay_violations,
};
pub use crate::deletion::DeleteVertexError;
pub use crate::io::visualization::{
    AdjacencyRecord, MESH_EXPORT_SCHEMA, MESH_EXPORT_SCHEMA_VERSION, MeshAdjacencyRecord,
    MeshExport, MeshExportError, MeshExportValidationError, MeshSimplexRecord, MeshVertexRecord,
    SimplexRecord, VISUALIZATION_SCHEMA, VISUALIZATION_SCHEMA_VERSION, ValidatedMeshExport,
    ValidatedVisualizationData, VertexRecord, VisualizationData, VisualizationDataValidationError,
    VisualizationExportError, VisualizationMetadata, VisualizationTopologyGuarantee,
    VisualizationTopologyKind,
};
pub use crate::repair::{
    DelaunayCheckPolicy, DelaunayRepairHeuristicConfig, DelaunayRepairHeuristicSeeds,
    DelaunayRepairOperation, DelaunayRepairOutcome, DelaunayRepairPolicy,
};
pub use crate::tds::{
    InvariantError, InvariantKind, InvariantViolation, TriangulationValidationReport,
};
pub use crate::triangulation::*;
pub use crate::validation::{
    DelaunayTriangulationValidationError, DelaunayVerificationError, DelaunayVerificationErrorKind,
};

/// Creates vertices from points by re-validating coordinates at the public boundary.
///
/// This helper is useful when point generators or parsing code have already
/// produced [`Point`](crate::geometry::Point) values, but the caller still wants
/// vertex construction to pass through the same fallible validation path as
/// [`tds::Vertex::try_new`]. New vertices have fresh UUIDs, no user data, and no
/// incident simplex pointer until inserted into a triangulation data structure.
///
/// # Errors
///
/// Returns [`geometry::CoordinateConversionError`] when any point coordinate
/// cannot be converted exactly to finite `f64`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     CoordinateConversionError, CoordinateValidationError, Point,
/// };
/// use delaunay::try_vertices_from_points;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Conversion(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Validation(#[from] CoordinateValidationError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let points = [Point::try_new([0.0, 0.0])?, Point::try_new([1.0, 0.0])?];
/// let vertices = try_vertices_from_points(&points)?;
/// assert_eq!(vertices.len(), 2);
/// # Ok(())
/// # }
/// ```
pub fn try_vertices_from_points<const D: usize>(
    points: &[geometry::Point<D>],
) -> Result<Vec<tds::Vertex<(), D>>, geometry::CoordinateConversionError> {
    points
        .iter()
        .map(|point| tds::Vertex::try_new(*point.coords()))
        .collect()
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
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
/// };
/// use delaunay::prelude::geometry::CoordinateConversionError;
/// use delaunay::prelude::topology::validation;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Topology(#[from] delaunay::topology::TopologyError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     vertex![0.0, 0.0, 0.0]?,
///     vertex![1.0, 0.0, 0.0]?,
///     vertex![0.0, 1.0, 0.0]?,
///     vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let result = validation::validate_triangulation_euler(dt.tds(), dt.global_topology())?;
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
        pub use global_topology_model::GlobalTopologyModelError;
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

    /// Ridge candidates, borrowed ridge queries, and lifted ridge-link views.
    pub mod ridge;

    /// Concrete topological space implementations.
    ///
    /// This module contains the currently exposed Euclidean, spherical, and
    /// toroidal space models that implement the topology traits used by
    /// construction policy and topology validation APIs.
    pub mod spaces {
        /// Euclidean space topology
        pub mod euclidean;
        /// Spherical space topology
        pub mod spherical;
        /// Toroidal space topology
        pub mod toroidal;

        pub use euclidean::EuclideanSpace;
        pub use spherical::SphericalSpace;
        pub use toroidal::{LiftedLinkEdge, LiftedVertexId, ToroidalSpace};
    }

    // Re-export commonly used types
    pub use crate::TopologyGuarantee;
    pub use characteristics::*;
    pub use manifold::{
        BoundaryFacetClassification, ManifoldError, classify_boundary_facet,
        validate_closed_boundary, validate_ridge_links, validate_vertex_links,
    };
    pub use ridge::{
        RidgeCandidate, RidgeCandidateError, RidgeLinkView, RidgeQuery, RidgeView,
        ridge_star_simplices,
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
/// counts.insert("simplices", 3);
///
/// let mut scratch: SmallBuffer<usize, 4> = SmallBuffer::new();
/// scratch.push(counts["simplices"]);
///
/// assert_eq!(scratch.as_slice(), &[3]);
/// ```
pub mod collections {
    pub use crate::core::collections::{
        Entry, FacetIndex, FacetIssuesMap, FacetSharingSimplicesBuffer, FacetVertexMap,
        FastBuildHasher, FastHashMap, FastHashSet, FastHasher, KeyBasedSimplexMap,
        KeyBasedVertexMap, MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, PeriodicOffsetBuffer,
        SecureHashMap, SecureHashSet, SimplexKeyBuffer, SimplexKeySet, SimplexNeighborsMap,
        SimplexSecondaryMap, SimplexToVertexUuidsMap, SimplexVertexBuffer, SimplexVertexKeyBuffer,
        SimplexVertexKeysMap, SimplexVertexUuidBuffer, SimplexVerticesMap, SmallBuffer, Uuid,
        UuidToSimplexKeyMap, UuidToVertexKeyMap, VertexKeyBuffer, VertexKeySet, VertexSecondaryMap,
        VertexToSimplicesMap, VertexUuidBuffer, VertexUuidSet, fast_hash_map_with_capacity,
        fast_hash_set_with_capacity, small_buffer_with_capacity_2, small_buffer_with_capacity_8,
        small_buffer_with_capacity_16,
    };

    /// Expert aliases for algorithm-local scratch buffers.
    ///
    /// These remain public for advanced users and APIs that expose exact buffer
    /// shapes, but they are separated from the common collection aliases to
    /// avoid accidental broad imports.
    pub mod algorithm_buffers {
        pub use crate::core::collections::{
            BadSimplexBuffer, CLEANUP_OPERATION_BUFFER_SIZE, CavityBoundaryBuffer, FacetInfoBuffer,
            GeometricPointBuffer, PointBuffer, SimplexRemovalBuffer, ValidSimplicesBuffer,
            ViolationBuffer,
        };
    }
}

/// Public low-level topology data structures and TDS helpers.
///
/// Use this module when you need simplices, facets, keys, the
/// [`Tds`](crate::tds::Tds) container, validation reports, or TDS-specific
/// helpers without reaching into the internal implementation namespace.
///
/// # Examples
///
/// ```rust
/// use delaunay::tds::Tds;
///
/// let tds: Tds<(), (), 2> = Tds::empty();
///
/// assert_eq!(tds.number_of_vertices(), 0);
/// assert_eq!(tds.number_of_simplices(), 0);
/// ```
pub mod tds {
    pub use crate::core::adjacency::*;
    pub use crate::core::collections::{
        FacetIndex, FastHashMap, FastHashSet, NeighborBuffer, PeriodicOffsetBuffer,
        SimplexKeyBuffer, SmallBuffer, Uuid,
    };
    pub use crate::core::edge::*;
    pub use crate::core::facet::*;
    pub use crate::core::simplex::*;
    pub use crate::core::tds::*;
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
/// blocks. Higher-level Delaunay construction, repair, and editing APIs are
/// available at the crate root and through the matching focused preludes.
///
/// # Examples
///
/// ```rust
/// use delaunay::algorithms::{LocateError, locate};
/// use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
/// use delaunay::tds::Tds;
///
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let tds: Tds<(), (), 2> = Tds::empty();
/// let kernel = AdaptiveKernel::new();
/// let point = Point::try_from([0.0, 0.0])?;
///
/// std::assert_matches!(
///     locate(&tds, &kernel, &point, None),
///     Err(LocateError::EmptyTriangulation)
/// );
/// # Ok(())
/// # }
/// ```
pub mod algorithms {
    #[cfg(any(feature = "diagnostics", all(test, debug_assertions)))]
    pub use crate::core::algorithms::locate::verify_conflict_region_completeness;
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
    pub use crate::core::query::QueryError;
    pub use crate::core::traits::data_type::{
        DataCopy, DataDebug, DataDeserialize, DataIdentity, DataSerde, DataSerialize, DataType,
    };
    pub use crate::core::traits::facet_incidence_analysis::FacetIncidenceAnalysis;
    pub use crate::core::util::{
        JaccardComputationError, extract_edge_set, extract_facet_identifier_set,
        extract_hull_facet_set, extract_vertex_coordinate_set, format_jaccard_report,
        jaccard_distance, jaccard_index, measure_with_result,
    };
    pub use crate::geometry::Point;
    pub use crate::geometry::algorithms::convex_hull::{
        ConvexHull, ConvexHullConstructionError, ConvexHullValidationError,
    };
    pub use crate::geometry::kernel::{
        AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel,
    };
    pub use crate::geometry::traits::coordinate::Coordinate;
    pub use crate::geometry::{insphere, insphere_distance, insphere_lifted};
    pub use crate::tds::{
        AllFacetsIter, BoundaryFacetsIter, EdgeIndex, EdgeKey, EdgeKeyError, EdgeView, FacetHandle,
        FacetIncidenceView, FacetToSimplicesIndex, FacetView, IncidenceView, OneSidedFacetsIter,
        Simplex, SimplexFacetsIter, SimplexKey, SimplexNeighborIndex, TopologyIndexBuildError,
        TriangulationAdjacency, Vertex, VertexKey,
    };
    pub use crate::{DelaunayTriangulation, Triangulation};
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export the public low-level facades.
    pub use crate::query::{
        DataCopy, DataDebug, DataDeserialize, DataIdentity, DataSerde, DataSerialize, DataType,
        FacetIncidenceAnalysis, QueryError,
    };
    pub use crate::tds::*;
    pub use crate::vertex;
    pub use crate::{
        ConstructionOptions, ConstructionSkipSample, ConstructionSlowInsertionSample,
        ConstructionStatistics, DedupPolicy, DedupTolerance, DelaunayCheckPolicy,
        DelaunayConstructionFailure, DelaunayConstructionRepairPhase,
        DelaunayConstructionRetryFailure, DelaunayError, DelaunayRepairHeuristicConfig,
        DelaunayRepairHeuristicSeeds, DelaunayRepairOperation, DelaunayRepairOutcome,
        DelaunayRepairPolicy, DelaunayResult, DelaunayTriangulation, DelaunayTriangulationBuilder,
        DelaunayTriangulationConstructionError,
        DelaunayTriangulationConstructionErrorWithStatistics, DelaunayTriangulationValidationError,
        DelaunayVerificationError, DelaunayVerificationErrorKind, DuplicateDetectionMetrics,
        FinalDelaunayValidationContext, FinalTopologyValidationContext, InitialSimplexStrategy,
        InsertionOrderStrategy, InsertionResult, PeriodicDomainPeriodError, PlManifoldRepairError,
        PlManifoldRepairStats, RepairDecision, RepairSkipReason, RetryPolicy, TopologicalOperation,
        TopologyGuarantee, Triangulation, TriangulationConstructionError,
        TriangulationEmbeddingIntersectionDetail, TriangulationEmbeddingSimplexDetail,
        TriangulationEmbeddingSimplexPairDetail, TriangulationEmbeddingValidationError,
        TriangulationEmbeddingValidationErrorKind, TriangulationEmbeddingValidationReport,
        TriangulationValidationError, TriangulationValidationReport, ValidationConfigurationError,
        ValidationPolicy, try_vertices_from_points,
    };

    // Re-export utility items, but avoid exporting the util module names themselves.
    //
    // In particular, exporting a local `uuid` module conflicts with the external `uuid`
    // crate name, making `use uuid::Uuid;` ambiguous for downstream users.
    pub use self::ordering::{
        HilbertBitDepth, HilbertError, HilbertQuantizedBatch, HilbertQuantizedVec,
        MAX_HILBERT_BITS, hilbert_index_in_range, hilbert_indices_for_quantized_batch,
        hilbert_indices_prequantized, hilbert_quantize_batch_in_range, hilbert_quantize_in_range,
        hilbert_sort_by_stable_in_range, hilbert_sort_by_unstable_in_range,
        hilbert_sorted_indices_in_range, try_hilbert_index, try_hilbert_quantize,
        try_hilbert_sort_by_stable, try_hilbert_sort_by_unstable, try_hilbert_sorted_indices,
    };
    pub use crate::core::util::{
        DeduplicationError, dedup_vertices_epsilon, dedup_vertices_exact,
        filter_vertices_excluding, try_dedup_vertices_epsilon,
    };
    pub use crate::delaunay_property_validation::{
        DelaunayValidationError, DelaunayViolationDetail, DelaunayViolationReport,
        delaunay_violation_report, find_delaunay_violations,
    };
    pub use crate::query::{
        JaccardComputationError, extract_edge_set, extract_facet_identifier_set,
        extract_hull_facet_set, extract_vertex_coordinate_set, format_jaccard_report,
        jaccard_distance, jaccard_index, measure_with_result,
    };
    pub use crate::tds::{
        UuidValidationError, checked_facet_key_from_vertex_keys, facet_view_to_vertices,
        facet_views_are_adjacent, make_uuid, stable_hash_u64_slice, usize_to_u8, validate_uuid,
        verify_facet_index_consistency,
    };
    pub use crate::topology::{
        GlobalTopology, GlobalTopologyModelError, TopologyError, TopologyKind,
        ToroidalConstructionMode, ToroidalDomain, ToroidalDomainError,
    };

    // Re-export point location algorithms from the public algorithms facade.
    pub use crate::algorithms::{
        ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
        LocateFallbackReason, LocateResult, LocateStats, locate, locate_with_stats,
    };

    // Re-export incremental insertion types
    pub use crate::{
        CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind,
        DelaunayRepairFailureContext, HullExtensionReason, InitialSimplexConstructionError,
        InitialSimplexUnexpectedInsertionStage, InsertionError, InsertionErrorKind,
        InsertionErrorSourceKind, InsertionTopologyValidationContext, NeighborRebuildError,
        NeighborWiringError, SpatialIndexConstructionFailure, TdsConstructionFailure,
        TdsValidationFailure,
    };
    pub use crate::{InsertionOutcome, InsertionStatistics, SuspicionFlags};

    // Re-export diagnostic types for scientific analysis of construction and repair
    pub use crate::flips::{
        DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairHeuristicRebuildFailure,
        DelaunayRepairHeuristicRebuildFailureKind, DelaunayRepairHeuristicVertexContext,
        DelaunayRepairOrientationCanonicalizationFailure,
        DelaunayRepairOrientationCanonicalizationFailureKind, DelaunayRepairPostconditionFailure,
        DelaunayRepairStats, DelaunayRepairVerificationContext, FlipContextError,
        FlipEdgeAdjacencyError, FlipError, FlipFailureKind, FlipMutationError,
        FlipNeighborCavityFailureKind, FlipNeighborDelaunayValidationFailureKind,
        FlipNeighborHullExtensionFailureKind, FlipNeighborRepairDiagnostics,
        FlipNeighborRepairFailure, FlipNeighborWiringError, FlipOrientationCheckStage,
        FlipPredicateError, FlipPredicateOperation, FlipTriangleAdjacencyError,
        FlipVertexAdjacencyError, RepairQueueOrder, TriangleHandleError,
    };

    // Re-export commonly used collection types from the public collections facade.
    // These are frequently used in advanced examples and downstream code
    pub use crate::collections::{
        FastHashMap, FastHashSet, SecureHashMap, SecureHashSet, SimplexNeighborsMap,
        SimplexSecondaryMap, SmallBuffer, VertexSecondaryMap, VertexToSimplicesMap,
        fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    };

    // Re-export from geometry
    pub use crate::geometry::{
        algorithms::*, coordinate_range::*, kernel::*, matrix::*, point::*, predicates::*,
        quality::*, robust_predicates::*, traits::coordinate::*, util::*,
    };

    /// Batch construction options, builders, and construction errors.
    ///
    /// This focused prelude is for callers configuring Delaunay construction
    /// without importing the broader triangulation editing and repair
    /// surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices)
    ///     .build::<()>()?;
    ///
    /// assert_eq!(triangulation.number_of_vertices(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub mod construction {
        pub use crate::builder::{DelaunayTriangulationBuilder, ExplicitConstructionError};
        pub use crate::construction::{
            ConstructionOptions, ConstructionSkipSample, ConstructionSlowInsertionSample,
            ConstructionStatistics, DedupPolicy, DedupTolerance, DelaunayConstructionFailure,
            DelaunayConstructionRepairPhase, DelaunayConstructionRetryFailure, DelaunayError,
            DelaunayResult, DelaunayTriangulationConstructionError,
            DelaunayTriangulationConstructionErrorWithStatistics, InitialSimplexStrategy,
            InsertionOrderStrategy, RetryPolicy,
        };
        pub use crate::core::util::DeduplicationError;
        pub use crate::geometry::coordinate_range::{
            CoordinateRangeBound, CoordinateRangeError, CoordinateRangeOrdering,
            InvalidCoordinateValue,
        };
        pub use crate::geometry::traits::coordinate::CoordinateValidationError;
        pub use crate::geometry::util::{InvalidPositiveScalar, RandomPointGenerationError};
        pub use crate::repair::DelaunayRepairPolicy;
        pub use crate::tds::{
            SimplexValidationError, SimplexValidationReport, Vertex, VertexValidationError,
            VertexValidationReport,
        };
        pub use crate::topology::traits::{
            GlobalTopology, GlobalTopologyModelError, TopologyKind, ToroidalConstructionMode,
            ToroidalDomain, ToroidalDomainError,
        };
        pub use crate::validation::{
            DelaunayTriangulationValidationError, DelaunayVerificationError,
            DelaunayVerificationErrorKind,
        };
        pub use crate::vertex;
        pub use crate::{
            CavityFillingError, CavityRepairStage, DelaunayTriangulation, DeleteVertexError,
            FinalDelaunayValidationContext, FinalTopologyValidationContext,
            SpatialIndexConstructionFailure, TopologyGuarantee, Triangulation,
            TriangulationConstructionError, try_vertices_from_points,
        };
    }

    /// Generic triangulation construction, validation, query, and local repair.
    ///
    /// This focused prelude is for callers working directly with
    /// [`Triangulation`] rather than the higher-level
    /// [`DelaunayTriangulation`] wrapper. It keeps the generic TDS/kernel/error
    /// types needed by public `Triangulation` methods together without pulling
    /// in Delaunay repair, delaunayize, or batch-construction APIs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::triangulation::{
    ///     FastKernel, Triangulation, TriangulationConstructionError, vertex,
    /// };
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Source(#[from] TriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let tds = Triangulation::<FastKernel<f64>, (), (), 2>::build_initial_simplex(&vertices)?;
    ///
    /// assert_eq!(tds.number_of_vertices(), 3);
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub mod triangulation {
        pub use crate::collections::{FacetIssuesMap, SimplexKeyBuffer, SmallBuffer};
        pub use crate::geometry::kernel::{
            AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel,
        };
        pub use crate::geometry::point::Point;
        pub use crate::query::{
            AllFacetsIter, BoundaryFacetsIter, DataCopy, DataDebug, DataDeserialize, DataIdentity,
            DataSerde, DataSerialize, DataType, EdgeIndex, EdgeKey, EdgeKeyError, EdgeView,
            FacetIncidenceAnalysis, FacetIncidenceView, FacetToSimplicesIndex, FacetView,
            IncidenceView, OneSidedFacetsIter, QueryError, SimplexFacetsIter, SimplexNeighborIndex,
            TopologyIndexBuildError, TriangulationAdjacency,
        };
        pub use crate::tds::{
            FacetHandle, InvariantError, NeighborSlot, Simplex, SimplexKey, Tds,
            TdsConstructionError, TdsError, TdsErrorKind, TdsMutationError,
            TriangulationValidationErrorKind, Vertex, VertexKey,
        };
        pub use crate::vertex;
        pub use crate::{
            InsertionError, PeriodicDomainPeriodError, SpatialIndexConstructionFailure,
            TopologyGuarantee, Triangulation, TriangulationConstructionError,
            TriangulationEmbeddingIntersectionDetail, TriangulationEmbeddingSimplexDetail,
            TriangulationEmbeddingSimplexPairDetail, TriangulationEmbeddingValidationError,
            TriangulationEmbeddingValidationErrorKind, TriangulationEmbeddingValidationReport,
            TriangulationValidationError, TriangulationValidationReport,
            ValidationConfigurationError, ValidationPolicy,
        };
    }

    /// Unified Pachner move workflow for Monte-Carlo-style local edits.
    ///
    /// This focused prelude exports the unified request/result/dispatch API,
    /// the handles and handle-construction errors needed to construct moves,
    /// result metadata needed to inspect moves, and [`vertex!`](crate::vertex)
    /// for k=1 insertion vertices. Low-level flip primitives stay under
    /// [`crate::flips`] for expert/debug workflows.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, TopologyGuarantee,
    /// };
    /// use delaunay::prelude::pachner::{
    ///     BistellarFlipKind, FlipDirection, PachnerMove, PachnerMoves, vertex,
    /// };
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0, 0.0]?,
    ///     vertex![1.0, 0.0, 0.0]?,
    ///     vertex![0.0, 1.0, 0.0]?,
    ///     vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices)
    ///     .topology_guarantee(TopologyGuarantee::PLManifold)
    ///     .build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    ///
    /// let result = dt.attempt_pachner(PachnerMove::K1Insert {
    ///     simplex_key,
    ///     vertex: vertex![0.2, 0.2, 0.2]?,
    /// })?;
    /// assert_eq!(result.kind, BistellarFlipKind::k1(3));
    /// assert_eq!(result.direction, FlipDirection::Forward);
    /// assert_eq!(result.inserted_face_vertices.len(), 1);
    /// assert_eq!(result.new_simplices.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The focused prelude intentionally excludes the lower-level flip trait:
    ///
    /// ```compile_fail
    /// use delaunay::prelude::pachner::BistellarFlips;
    /// ```
    ///
    /// Unified Pachner imports do not expose the primitive flip methods:
    ///
    /// ```compile_fail
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder};
    /// use delaunay::prelude::pachner::*;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0, 0.0]?,
    ///     vertex![1.0, 0.0, 0.0]?,
    ///     vertex![0.0, 1.0, 0.0]?,
    ///     vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let mut dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let Some((simplex_key, _)) = dt.simplices().next() else {
    ///     return Ok(());
    /// };
    /// let Ok(facet) = FacetHandle::try_new(dt.tds(), simplex_key, 0) else {
    ///     return Ok(());
    /// };
    /// if dt.flip_k2(facet).is_err() {
    ///     return Ok(());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub mod pachner {
        pub use crate::flips::{
            BistellarFlipKind, FlipDirection, FlipError, RidgeHandle, TriangleHandle,
            TriangleHandleError,
        };
        pub use crate::pachner::{PachnerMove, PachnerMoveResult, PachnerMoves};
        pub use crate::tds::{
            EdgeKey, EdgeKeyError, FacetError, FacetHandle, SimplexKey, Vertex, VertexKey,
        };
        pub use crate::vertex;
    }

    /// Incremental insertion building blocks and diagnostics.
    pub mod insertion {
        pub use crate::collections::SimplexKeyBuffer;
        pub use crate::tds::FacetHandle;
        pub use crate::tds::{SimplexKey, Tds, TdsMutationError, VertexKey};
        pub use crate::{
            CavityFillingError, CavityRepairStage, DelaunayRepairErrorKind,
            DelaunayRepairFailureContext, HullExtensionReason, InitialSimplexConstructionError,
            InitialSimplexUnexpectedInsertionStage, InsertionError, InsertionErrorKind,
            InsertionErrorSourceKind, InsertionTopologyValidationContext, NeighborRebuildError,
            NeighborWiringError, SpatialIndexConstructionFailure, TdsConstructionFailure,
            TdsValidationFailure, extend_hull, fill_cavity, repair_neighbor_pointers,
            repair_neighbor_pointers_local, wire_cavity_neighbors,
        };
        pub use crate::{InsertionOutcome, InsertionResult, InsertionStatistics};
    }

    /// Vertex deletion errors and key types.
    ///
    /// Deletion itself is an inherent method on
    /// [`DelaunayTriangulation`], so callers
    /// usually import the triangulation type from
    /// [`prelude::construction`](crate::prelude::construction) and import this
    /// focused prelude only when they need to match deletion-specific failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::deletion::{DeleteVertexError, VertexKey};
    /// use slotmap::KeyData;
    ///
    /// let err = DeleteVertexError::VertexNotFound {
    ///     vertex_key: VertexKey::from(KeyData::from_ffi(42)),
    /// };
    /// std::assert_matches!(err, DeleteVertexError::VertexNotFound { .. });
    /// ```
    pub mod deletion {
        pub use crate::DeleteVertexError;
        pub use crate::tds::VertexKey;
    }

    /// Topological operation telemetry and repair decisions.
    pub mod operations {
        pub use crate::{
            InsertionOutcome, InsertionResult, InsertionStatistics, RepairDecision,
            RepairSkipReason, SuspicionFlags, TopologicalOperation,
        };
    }

    /// Flip-based Delaunay repair, diagnostics, and Level 5 validation.
    ///
    /// ```rust
    /// use delaunay::prelude::repair::{
    ///     FlipNeighborHullExtensionFailureKind, FlipNeighborRepairFailure,
    /// };
    ///
    /// let reason = FlipNeighborHullExtensionFailureKind::DisconnectedVisiblePatch;
    /// std::assert_matches!(
    ///     reason,
    ///     FlipNeighborHullExtensionFailureKind::DisconnectedVisiblePatch
    /// );
    ///
    /// let _ = std::mem::size_of::<FlipNeighborRepairFailure>();
    /// ```
    pub mod repair {
        pub use crate::flips::{
            DelaunayRepairDiagnostics, DelaunayRepairError, DelaunayRepairHeuristicRebuildFailure,
            DelaunayRepairHeuristicRebuildFailureKind, DelaunayRepairHeuristicVertexContext,
            DelaunayRepairOrientationCanonicalizationFailure,
            DelaunayRepairOrientationCanonicalizationFailureKind,
            DelaunayRepairPostconditionFailure, DelaunayRepairStats,
            DelaunayRepairVerificationContext, FlipContextError, FlipEdgeAdjacencyError, FlipError,
            FlipFailureKind, FlipMutationError, FlipNeighborCavityFailureKind,
            FlipNeighborDelaunayValidationFailureKind, FlipNeighborHullExtensionFailureKind,
            FlipNeighborRepairDiagnostics, FlipNeighborRepairFailure, FlipNeighborWiringError,
            FlipOrientationCheckStage, FlipPredicateError, FlipPredicateOperation,
            FlipTriangleAdjacencyError, FlipVertexAdjacencyError, RepairQueueOrder,
            TriangleHandleError, verify_delaunay_for_triangulation,
            verify_delaunay_via_flip_predicates,
        };
        pub use crate::repair::{
            DelaunayCheckPolicy, DelaunayRepairHeuristicConfig, DelaunayRepairHeuristicSeeds,
            DelaunayRepairOutcome, DelaunayRepairPolicy,
        };
        pub use crate::{
            DelaunayRepairErrorKind, DelaunayRepairOperation, DelaunayTriangulation,
            DelaunayTriangulationValidationError, DelaunayVerificationError,
            DelaunayVerificationErrorKind,
        };
        pub use crate::{
            DelaunayValidationError, DelaunayViolationDetail, DelaunayViolationReport,
            delaunay_violation_report, find_delaunay_violations,
        };
        pub use crate::{
            TopologyGuarantee, Triangulation, ValidationConfigurationError, ValidationPolicy,
        };
    }

    /// End-to-end "repair then delaunayize" workflow.
    ///
    /// Self-contained: a single `use delaunay::prelude::delaunayize::*`
    /// import brings in [`DelaunayTriangulationBuilder`], [`DelaunayTriangulation`],
    /// PL-manifold repair types, and all delaunayize-specific types.
    pub mod delaunayize {
        pub use crate::delaunayize::*;
        pub use crate::{DelaunayTriangulation, DelaunayTriangulationBuilder};
        pub use crate::{PlManifoldRepairError, PlManifoldRepairStats};
    }

    /// Delaunay-level validation APIs, reports, and construction diagnostics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::validation::ValidationCadence;
    ///
    /// let cadence = ValidationCadence::from_optional_every(Some(32));
    /// assert!(!cadence.should_validate(31));
    /// assert!(cadence.should_validate(32));
    /// ```
    pub mod validation {
        pub use crate::validation::*;
        pub use crate::{
            DelaunayTriangulationValidationError, DelaunayVerificationError,
            DelaunayVerificationErrorKind, PeriodicDomainPeriodError, TopologyGuarantee,
            TriangulationEmbeddingIntersectionDetail, TriangulationEmbeddingSimplexDetail,
            TriangulationEmbeddingSimplexPairDetail, TriangulationEmbeddingValidationError,
            TriangulationEmbeddingValidationErrorKind, TriangulationEmbeddingValidationReport,
            TriangulationValidationError, TriangulationValidationReport,
            ValidationConfigurationError, ValidationPolicy,
        };
        pub use crate::{
            DelaunayValidationError, DelaunayViolationDetail, DelaunayViolationReport,
            delaunay_violation_report, find_delaunay_violations,
        };
    }

    /// Focused exports for collection types used throughout the crate.
    ///
    /// This prelude keeps common map, set, key-map, and small-buffer aliases
    /// convenient without importing every algorithm-specific scratch buffer.
    /// Expert-only buffers remain available from [`crate::collections`]
    /// or the nested [`crate::prelude::collections::algorithm_buffers`] module.
    ///
    /// ```compile_fail
    /// use delaunay::prelude::collections::SimplexRemovalBuffer;
    /// ```
    pub mod collections {
        pub use crate::collections::{
            Entry, FacetIndex, FacetIssuesMap, FacetSharingSimplicesBuffer, FastBuildHasher,
            FastHashMap, FastHashSet, FastHasher, KeyBasedSimplexMap, KeyBasedVertexMap,
            MAX_PRACTICAL_DIMENSION_SIZE, NeighborBuffer, PeriodicOffsetBuffer, SecureHashMap,
            SecureHashSet, SimplexKeyBuffer, SimplexKeySet, SimplexNeighborsMap,
            SimplexSecondaryMap, SimplexToVertexUuidsMap, SimplexVertexBuffer,
            SimplexVertexKeyBuffer, SimplexVertexKeysMap, SimplexVertexUuidBuffer,
            SimplexVerticesMap, SmallBuffer, Uuid, UuidToSimplexKeyMap, UuidToVertexKeyMap,
            VertexKeyBuffer, VertexKeySet, VertexSecondaryMap, VertexToSimplicesMap,
            VertexUuidBuffer, VertexUuidSet, fast_hash_map_with_capacity,
            fast_hash_set_with_capacity, small_buffer_with_capacity_2,
            small_buffer_with_capacity_8, small_buffer_with_capacity_16,
        };

        /// Expert aliases for algorithm-local scratch buffers.
        ///
        /// These remain public for advanced users and for APIs that expose their
        /// exact buffer shapes, but they are separated from the common
        /// collections prelude to avoid accidental broad imports.
        pub mod algorithm_buffers {
            pub use crate::collections::algorithm_buffers::{
                BadSimplexBuffer, CLEANUP_OPERATION_BUFFER_SIZE, CavityBoundaryBuffer,
                FacetInfoBuffer, GeometricPointBuffer, PointBuffer, SimplexRemovalBuffer,
                ValidSimplicesBuffer, ViolationBuffer,
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
    /// let tds: Tds<(), (), 2> = Tds::empty();
    ///
    /// assert_eq!(tds.number_of_vertices(), 0);
    /// assert_eq!(tds.number_of_simplices(), 0);
    /// ```
    pub mod tds {
        pub use crate::collections::{
            FacetIndex, FastHashMap, FastHashSet, NeighborBuffer, PeriodicOffsetBuffer,
            SimplexKeyBuffer, SmallBuffer, Uuid,
        };
        pub use crate::tds::*;
    }

    /// Focused exports for geometry types, predicates, and helpers.
    pub mod geometry {
        pub use crate::geometry::{
            coordinate_range::{
                CoordinateRange, CoordinateRangeBound, CoordinateRangeError,
                CoordinateRangeOrdering, InvalidCoordinateValue,
            },
            kernel::{AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel},
            matrix::{LaError, Matrix, MatrixError, determinant},
            point::Point,
            predicates::{
                InSphere, Orientation, insphere, insphere_distance, insphere_lifted,
                simplex_orientation,
            },
            quality::{
                QualityDegeneracyMeasure, QualityError, QualityNumericOperation,
                QualitySimplexVerticesError, normalized_volume, radius_ratio,
            },
            robust_predicates::{
                ConsistencyResult, InsphereConsistencyError, robust_insphere, robust_orientation,
            },
            traits::coordinate::{
                Coordinate, CoordinateConversionError, CoordinateConversionValue,
                CoordinateIdentity, CoordinateRepresentation, CoordinateValidationError,
                CoordinateValues, DEFAULT_TOLERANCE_F64, DegenerateSimplexReason,
                F64_MANTISSA_DIGITS, FiniteCheck, FiniteCoordinateValue, HashCoordinate,
                OrderedCmp, OrderedEq,
            },
            util::{
                ArrayConversionFailureReason, CircumcenterError, CircumcenterFailureReason,
                DegenerateGeometry, DegenerateMeasure, SurfaceMeasureError, ValueConversionError,
                ValueConversionFailureReason, circumcenter, circumradius, circumradius_with_center,
                facet_measure, hypot, inradius, safe_coords_from_f64, safe_coords_to_f64,
                safe_scalar_from_f64, safe_scalar_to_f64, safe_usize_to_scalar, simplex_volume,
                squared_norm, surface_measure,
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
    /// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
    /// let tds: Tds<(), (), 2> = Tds::empty();
    /// let kernel = AdaptiveKernel::new();
    /// let point = Point::try_from([0.0, 0.0])?;
    ///
    /// std::assert_matches!(
    ///     locate(&tds, &kernel, &point, None),
    ///     Err(LocateError::EmptyTriangulation)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub mod algorithms {
        pub use crate::algorithms::{
            ConflictError, InternalInconsistencySite, LocateError, LocateFallback,
            LocateFallbackReason, LocateResult, LocateStats, extract_cavity_boundary,
            find_conflict_region, locate, locate_with_stats,
        };
    }

    /// Focused exports for construction telemetry and opt-in diagnostic helpers.
    ///
    /// Construction telemetry is always available.  Expensive verification and
    /// violation-report helpers are compiled only with the `diagnostics`
    /// feature because they are intended for explicit debugging workflows, not
    /// the default public API surface.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::diagnostics::NeighborSlot;
    ///
    /// assert!(NeighborSlot::Boundary.is_boundary());
    /// ```
    pub mod diagnostics {
        #[cfg(feature = "diagnostics")]
        #[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
        pub use crate::algorithms::verify_conflict_region_completeness;
        #[cfg(feature = "diagnostics")]
        #[cfg_attr(docsrs, doc(cfg(feature = "diagnostics")))]
        pub use crate::debug_print_first_delaunay_violation;
        pub use crate::diagnostics::{
            BatchLocalRepairTrigger, ConstructionTelemetry, LocalRepairSample,
        };
        pub use crate::tds::NeighborSlot;
        pub use crate::{
            DelaunayViolationDetail, DelaunayViolationReport, delaunay_violation_report,
        };
    }

    /// Focused exports for generic simplicial-complex export data.
    ///
    /// These records are intended for notebooks, visualization tools, ML
    /// pipelines, and downstream crates that want stable vertex/simplex ids
    /// without depending on storage-local TDS handles.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{
    ///     DelaunayResult, DelaunayTriangulationBuilder, vertex,
    /// };
    /// use delaunay::prelude::export::MESH_EXPORT_SCHEMA;
    ///
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
    /// let export = triangulation.to_mesh_export()?;
    ///
    /// assert_eq!(export.metadata.schema, MESH_EXPORT_SCHEMA);
    /// # Ok(())
    /// # }
    /// ```
    pub mod export {
        pub use crate::geometry::traits::coordinate::InvalidCoordinateValue;
        pub use crate::io::visualization::{
            AdjacencyRecord, MESH_EXPORT_SCHEMA, MESH_EXPORT_SCHEMA_VERSION, MeshAdjacencyRecord,
            MeshExport, MeshExportError, MeshExportValidationError, MeshSimplexRecord,
            MeshVertexRecord, SimplexRecord, VISUALIZATION_SCHEMA, VISUALIZATION_SCHEMA_VERSION,
            ValidatedMeshExport, ValidatedVisualizationData, VertexRecord, VisualizationData,
            VisualizationDataValidationError, VisualizationExportError, VisualizationMetadata,
            VisualizationTopologyGuarantee, VisualizationTopologyKind,
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
    ///   [`DelaunayTriangulation::simplex_neighbors`]
    /// - Fast repeated queries: [`DelaunayTriangulation::incidence`], [`DelaunayTriangulation::build_edge_index`],
    ///   [`DelaunayTriangulation::build_simplex_neighbor_index`], and composite
    ///   [`DelaunayTriangulation::adjacency`] / [`TriangulationAdjacency`]
    /// - Zero-allocation geometry accessors: [`DelaunayTriangulation::vertex_coords`],
    ///   [`DelaunayTriangulation::simplex_vertices`]
    /// - Convex hull extraction: [`ConvexHull::try_from_triangulation`]
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
        pub use crate::tds::{
            EdgeIndex, EdgeKey, EdgeKeyError, EdgeView, FacetHandle, FacetIncidenceView,
            FacetToSimplicesIndex, IncidenceView, SimplexKey, SimplexNeighborIndex,
            TopologyIndexBuildError, TriangulationAdjacency, VertexKey,
        };
        pub use crate::{DelaunayTriangulation, Triangulation};

        // Common input/output types (kept intentionally small)
        pub use crate::geometry::Point;
        pub use crate::geometry::kernel::{
            AdaptiveKernel, ExactPredicates, FastKernel, Kernel, RobustKernel,
        };
        pub use crate::geometry::traits::coordinate::Coordinate;
        pub use crate::query::{
            AllFacetsIter, BoundaryFacetsIter, DataCopy, DataDebug, DataDeserialize, DataIdentity,
            DataSerde, DataSerialize, DataType, FacetIncidenceAnalysis, FacetView,
            OneSidedFacetsIter, QueryError, Simplex, SimplexFacetsIter, Vertex,
        };

        // Read-only predicates (useful in benchmarks / lightweight geometry checks)
        pub use crate::geometry::{insphere, insphere_distance, insphere_lifted};

        // Read-only algorithms
        pub use crate::assert_jaccard_gte;
        pub use crate::geometry::algorithms::convex_hull::{
            ConvexHull, ConvexHullConstructionError, ConvexHullValidationError,
        };
        pub use crate::query::{
            JaccardComputationError, extract_edge_set, extract_facet_identifier_set,
            extract_hull_facet_set, extract_vertex_coordinate_set, format_jaccard_report,
            jaccard_distance, jaccard_index,
        };

        // Instrumentation helpers (no-op unless features enable extra tracking)
        pub use crate::query::measure_with_result;
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
    ///     CoordinateRange, RandomPointGenerationError, generate_random_points_in_range_seeded,
    /// };
    /// use delaunay::prelude::geometry::Point;
    ///
    /// # fn main() -> Result<(), RandomPointGenerationError> {
    /// let range = CoordinateRange::try_new(0.0_f64, 1.0)?;
    /// let points: Vec<Point<3>> =
    ///     generate_random_points_in_range_seeded(4, range, 42)?;
    ///
    /// assert_eq!(points.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub mod generators {
        pub use crate::TopologyGuarantee;
        pub use crate::construction::InsertionOrderStrategy;
        pub use crate::geometry::coordinate_range::{
            CoordinateRange, CoordinateRangeBound, CoordinateRangeError, CoordinateRangeOrdering,
            InvalidCoordinateValue,
        };
        pub use crate::geometry::util::{
            InvalidPositiveScalar, RandomPointGenerationError, RandomTriangulationBuilder,
            generate_grid_points, generate_poisson_points_in_range, generate_random_points_in_ball,
            generate_random_points_in_ball_seeded, generate_random_points_in_range,
            generate_random_points_in_range_seeded, generate_random_points_periodic,
            generate_random_triangulation_in_range,
            generate_random_triangulation_in_range_with_topology_guarantee,
            scaled_bounds_by_point_count, try_generate_poisson_points, try_generate_random_points,
            try_generate_random_points_seeded, try_generate_random_triangulation,
            try_generate_random_triangulation_with_topology_guarantee,
        };
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
    /// use delaunay::prelude::ordering::{HilbertBitDepth, HilbertError, try_hilbert_sorted_indices};
    ///
    /// let coords = [[0.9_f64, 0.9], [0.1, 0.1], [0.5, 0.5]];
    /// let bits = HilbertBitDepth::try_new(8)?;
    /// let order = try_hilbert_sorted_indices(&coords, (0.0, 1.0), bits)?;
    ///
    /// assert_eq!(order.len(), coords.len());
    /// # Ok::<(), HilbertError>(())
    /// ```
    pub mod ordering {
        pub use crate::core::util::{
            HilbertBitDepth, HilbertError, HilbertQuantizedBatch, HilbertQuantizedVec,
            MAX_HILBERT_BITS, hilbert_index_in_range, hilbert_indices_for_quantized_batch,
            hilbert_indices_prequantized, hilbert_quantize_batch_in_range,
            hilbert_quantize_in_range, hilbert_sort_by_stable_in_range,
            hilbert_sort_by_unstable_in_range, hilbert_sorted_indices_in_range, try_hilbert_index,
            try_hilbert_quantize, try_hilbert_sort_by_stable, try_hilbert_sort_by_unstable,
            try_hilbert_sorted_indices,
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
                BoundaryFacetClassification, ManifoldError, classify_boundary_facet,
                validate_closed_boundary, validate_ridge_links, validate_ridge_links_for_simplices,
                validate_vertex_links,
            };
            pub use crate::topology::ridge::{
                RidgeCandidate, RidgeCandidateError, RidgeLinkView, RidgeQuery, RidgeView,
                ridge_star_simplices,
            };
            pub use crate::topology::traits::{
                GlobalTopology, GlobalTopologyModelError, TopologicalSpace, TopologyError,
                TopologyKind, ToroidalConstructionMode,
            };
        }

        /// Topological space models and traits.
        pub mod spaces {
            pub use crate::topology::spaces::*;
            pub use crate::topology::traits::{
                GlobalTopology, GlobalTopologyModelError, TopologicalSpace, TopologyError,
                TopologyKind, ToroidalConstructionMode, ToroidalDomain, ToroidalDomainError,
            };
        }
    }
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
    use crate::geometry::matrix::LaError;
    use crate::{
        DelaunayTriangulation,
        core::{
            adjacency::TriangulationAdjacency, edge::EdgeKey, simplex::Simplex, tds::Tds,
            triangulation::Triangulation, vertex::Vertex,
        },
        geometry::{
            Point, algorithms::convex_hull::ConvexHull, kernel::AdaptiveKernel, kernel::FastKernel,
            util::CircumcenterError,
        },
        is_normal,
        prelude::delaunayize::{
            DelaunayTriangulationConstructionError, DelaunayizeConfig, DelaunayizeError,
            DelaunayizeOutcome, PlManifoldRepairError, PlManifoldRepairStats,
            SimplexDataRestoreError, SimplexValidationError,
        },
        prelude::repair::{
            DelaunayCheckPolicy, DelaunayRepairError, DelaunayRepairOutcome, DelaunayRepairPolicy,
            DelaunayRepairStats, DelaunayTriangulation as RepairDelaunayTriangulation,
            FlipContextError, FlipError, RepairQueueOrder, TopologyGuarantee,
            verify_delaunay_for_triangulation, verify_delaunay_via_flip_predicates,
        },
        prelude::*,
    };
    use std::assert_matches;

    #[cfg(feature = "count-allocations")]
    use allocation_counter::measure;

    // =============================================================================
    // TYPE SAFETY TESTS
    // =============================================================================

    #[test]
    fn normal_types() {
        assert!(is_normal::<Point<3>>());
        assert!(is_normal::<Vertex<(), 3>>());
        assert!(is_normal::<Simplex<(), 4>>());
        assert!(is_normal::<Tds<(), (), 4>>());
        assert!(is_normal::<Triangulation<FastKernel<f64>, (), (), 3>>());
        assert!(is_normal::<DelaunayTriangulation<FastKernel<f64>, (), (), 3>>());
        assert!(is_normal::<ConvexHull<(), (), 3>>());
        assert!(is_normal::<EdgeKey>());
        assert!(is_normal::<TriangulationAdjacency<'static>>());
        assert!(is_normal::<DelaunayizeConfig>());
        assert!(is_normal::<DelaunayizeOutcome<(), (), 3>>());
        assert!(is_normal::<DelaunayizeError>());
        assert!(is_normal::<DelaunayRepairError>());
        assert!(is_normal::<DelaunayRepairStats>());
        assert!(is_normal::<PlManifoldRepairError>());
        assert!(is_normal::<PlManifoldRepairStats<(), (), 3>>());
        assert!(is_normal::<SimplexDataRestoreError>());
        assert!(is_normal::<SimplexValidationError>());
        assert!(is_normal::<DelaunayError>());
        assert!(is_normal::<DelaunayTriangulationConstructionError>());
    }

    #[test]
    fn circumcenter_error_clones_linear_algebra_source() {
        let source = LaError::NonFinite {
            row: Some(1),
            col: 2,
        };
        let error = CircumcenterError::LinearAlgebraFailure { source };

        assert_eq!(error.clone(), error);
        assert!(error.to_string().contains("Linear algebra"));
    }

    #[test]
    fn la_errors_map_to_public_circumcenter_errors() {
        let unsupported = CircumcenterError::from(LaError::UnsupportedDimension {
            requested: 9,
            max: 7,
        });
        assert_eq!(
            unsupported,
            CircumcenterError::UnsupportedMatrixDimension {
                requested: 9,
                max: 7,
            }
        );

        let index_error = CircumcenterError::from(LaError::IndexOutOfBounds {
            row: 3,
            col: 4,
            dim: 2,
        });
        assert_eq!(
            index_error,
            CircumcenterError::MatrixError {
                source: MatrixError::OutOfBounds {
                    row: 3,
                    column: 4,
                    dimension: 2,
                },
            }
        );
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

        // Test domain-specific public types can be instantiated
        let _neighbors: SimplexNeighborsMap = SimplexNeighborsMap::default();
        let _vertex_simplices: VertexToSimplicesMap = VertexToSimplicesMap::default();
    }

    #[test]
    fn prelude_repair_exports() {
        let vertices = vec![
            crate::vertex![0.0, 0.0].unwrap(),
            crate::vertex![1.0, 0.0].unwrap(),
            crate::vertex![0.0, 1.0].unwrap(),
        ];
        let dt: RepairDelaunayTriangulation<_, (), (), 2> =
            RepairDelaunayTriangulation::try_new(&vertices).unwrap();
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
        assert_matches!(order, RepairQueueOrder::Fifo);
        assert_eq!(
            DelaunayRepairPolicy::default(),
            DelaunayRepairPolicy::EveryInsertion
        );
        assert_eq!(DelaunayCheckPolicy::default(), DelaunayCheckPolicy::EndOnly);

        let err = DelaunayRepairError::from(FlipError::DegenerateSimplex);
        assert_matches!(err, DelaunayRepairError::Flip { .. });
        let context_err = FlipContextError::ReplacementPeriodicOffsetCountMismatch {
            simplex_count: 1,
            offset_count: 0,
        };
        assert_matches!(
            context_err,
            FlipContextError::ReplacementPeriodicOffsetCountMismatch { .. }
        );
        let topo = TopologyGuarantee::PLManifold;
        assert_matches!(topo, TopologyGuarantee::PLManifold);
    }

    #[test]
    fn prelude_quality_exports() {
        // Test that quality functions are accessible from prelude
        let vertices = vec![
            crate::vertex![0.0, 0.0].unwrap(),
            crate::vertex![1.0, 0.0].unwrap(),
            crate::vertex![0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Get a simplex to test quality functions
        let (simplex_key, _) = dt.simplices().next().unwrap();

        // Test that quality functions are accessible
        let ratio = radius_ratio(dt.as_triangulation(), simplex_key).unwrap();
        assert!(ratio > 0.0);

        let norm_vol = normalized_volume(dt.as_triangulation(), simplex_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    #[test]
    fn test_prelude_kernel_exports() {
        // Test that kernel types and predicates are accessible from prelude
        let fast_kernel = FastKernel::<f64>::new();
        let robust_kernel = RobustKernel::<f64>::new();

        // Test 2D orientation predicate
        let triangle = [
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            fast_kernel.orientation(&collinear).unwrap(),
            0,
            "Collinear points should have zero orientation"
        );

        // Test in_sphere predicate
        let inside_point = Point::try_new([0.25, 0.25]).expect("finite point coordinates");
        let result = fast_kernel.in_sphere(&triangle, &inside_point).unwrap();
        assert_eq!(result, 1, "Point should be inside circumcircle");

        let outside_point = Point::try_new([2.0, 2.0]).expect("finite point coordinates");
        let result = fast_kernel.in_sphere(&triangle, &outside_point).unwrap();
        assert_eq!(result, -1, "Point should be outside circumcircle");
    }

    #[test]
    fn test_prelude_core_types() {
        // Test that core types are accessible and work from prelude
        // Point construction
        let p1 = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        let p2 = Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates");
        assert_ne!(p1, p2);

        // Vertex construction via the fallible smart constructor.
        let v1: Vertex<(), 3> = Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap();
        let v2: Vertex<(), 3> = Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap();
        assert_ne!(v1.point(), v2.point());

        // DelaunayTriangulation construction
        let vertices = vec![
            crate::vertex![0.0, 0.0, 0.0].unwrap(),
            crate::vertex![1.0, 0.0, 0.0].unwrap(),
            crate::vertex![0.0, 1.0, 0.0].unwrap(),
            crate::vertex![0.0, 0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        assert_eq!(dt.number_of_vertices(), 4);
        assert_eq!(dt.number_of_simplices(), 1);

        // Access Triangulation, Tds, Simplex types
        let tri = dt.as_triangulation();
        assert_eq!(tri.number_of_vertices(), 4);

        let tds = &tri.tds;
        assert_eq!(tds.number_of_simplices(), 1);

        // Iterate over simplices
        for (simplex_key, _simplex) in tri.simplices() {
            assert!(tds.simplex(simplex_key).is_some());
        }
    }

    #[test]
    fn test_prelude_point_location() {
        // Test that point location algorithms are accessible
        let vertices = vec![
            crate::vertex![0.0, 0.0].unwrap(),
            crate::vertex![1.0, 0.0].unwrap(),
            crate::vertex![0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // Test locate function with kernel
        let kernel = FastKernel::<f64>::new();
        let query_point = Point::try_new([0.3, 0.3]).expect("finite point coordinates");
        let result = locate(dt.tds(), &kernel, &query_point, None);
        assert!(result.is_ok());

        // Result should be a LocateResult
        match result.unwrap() {
            LocateResult::InsideSimplex(_)
            | LocateResult::OnFacet { .. }
            | LocateResult::OnEdge { .. }
            | LocateResult::OnVertex(_) => { /* expected or acceptable */ }
            LocateResult::Outside => panic!("Point should be inside triangulation"),
        }

        // Test outside point
        let outside_point = Point::try_new([10.0, 10.0]).expect("finite point coordinates");
        let result = locate(dt.tds(), &kernel, &outside_point, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prelude_geometry_types() {
        // Test Point with Coordinate trait
        let p = Point::try_new([1.0_f64, 2.0_f64, 3.0_f64]).expect("finite point coordinates");
        assert!((p.coords()[0] - 1.0_f64).abs() < f64::EPSILON);
        assert!((p.coords()[1] - 2.0_f64).abs() < f64::EPSILON);
        assert!((p.coords()[2] - 3.0_f64).abs() < f64::EPSILON);

        // Test predicates are accessible
        let triangle = [
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        // simplex_orientation is exported from predicates
        let orientation = simplex_orientation(&triangle).unwrap();
        assert_ne!(orientation, Orientation::DEGENERATE);

        // Test insphere predicate
        let test_point = Point::try_new([0.25, 0.25]).expect("finite point coordinates");
        let result = insphere(&triangle, test_point).unwrap();
        assert_eq!(result, InSphere::INSIDE);
    }

    #[test]
    fn test_prelude_convex_hull() {
        // Test that convex hull operations are accessible
        let vertices = vec![
            crate::vertex![0.0, 0.0, 0.0].unwrap(),
            crate::vertex![1.0, 0.0, 0.0].unwrap(),
            crate::vertex![0.0, 1.0, 0.0].unwrap(),
            crate::vertex![0.0, 0.0, 1.0].unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        // ConvexHull type should be accessible
        let hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).unwrap();
        assert_eq!(hull.number_of_facets(), 4); // Tetrahedron has 4 faces

        // Test point visibility
        let outside_point = Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates");
        let is_outside = hull
            .is_point_outside(&outside_point, dt.as_triangulation())
            .unwrap();
        assert!(is_outside);

        let inside_point = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");
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
