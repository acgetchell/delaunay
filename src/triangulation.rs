//! Triangulation-facing APIs.
//!
//! This module is the public facade for triangulation workflows.  It deliberately
//! stays thin:
//!
//! - [`crate::prelude::triangulation::Triangulation`] owns the generic
//!   triangulation container and low-level mutation invariants.
//! - [`crate::triangulation`] owns higher-level construction, Delaunay repair,
//!   diagnostics, validation scheduling, editing, and builder workflows.
//! - Submodules under this namespace keep those concerns separate while this
//!   facade preserves the stable public import surface.
//!
//! # Examples
//!
//! ```rust
//! use delaunay::prelude::triangulation::construction::{
//!     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
//! };
//!
//! # fn main() -> Result<(), DelaunayTriangulationConstructionError> {
//! let vertices = vec![
//!     vertex!([0.0, 0.0]),
//!     vertex!([1.0, 0.0]),
//!     vertex!([0.0, 1.0]),
//! ];
//! let triangulation = DelaunayTriangulationBuilder::new(&vertices)
//!     .build::<()>()?;
//!
//! assert_eq!(triangulation.number_of_vertices(), 3);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

/// Fluent builder for Delaunay triangulations.
///
/// See [`DelaunayTriangulation`](crate::triangulation::delaunay::DelaunayTriangulation)
/// for the constructed triangulation type.
pub mod builder;
/// Delaunay triangulation layer with incremental insertion.
pub mod delaunay;
/// End-to-end "repair then delaunayize" workflow.
pub mod delaunayize;
/// Construction and performance diagnostics.
pub mod diagnostics;
/// Triangulation editing operations (bistellar flips).
pub mod flips;
pub(crate) mod locality;
/// Validation scheduling helpers for triangulation diagnostics.
pub mod validation;

// Re-export commonly used triangulation types for discoverability.
pub use crate::core::triangulation::Triangulation;
pub use crate::triangulation::builder::DelaunayTriangulationBuilder;
pub use crate::triangulation::delaunay::DelaunayTriangulation;
