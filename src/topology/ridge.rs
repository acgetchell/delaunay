//! Ridge candidates, queries, views, and lifted ridge-link views.
//!
//! Ridges are codimension-2 simplices. Unlike vertices and D-simplices, they
//! are not stored directly in the TDS, so this module separates detached
//! [`RidgeCandidate`] values from borrowed [`RidgeQuery`] and [`RidgeView`]
//! values over one live triangulation. Periodic ridge links preserve toroidal
//! covering-space identity with
//! [`LiftedVertexId`](crate::topology::spaces::toroidal::LiftedVertexId) and
//! [`LiftedLinkEdge`](crate::topology::spaces::toroidal::LiftedLinkEdge).
//!
//! Ridge-star and ridge-link validation is combinatorial PL topology: it
//! follows standard links-of-simplices criteria for PL manifolds (see
//! `REFERENCES.md`, "Topological Manifolds and PL Topology") and does not add
//! any new `f64` floating-point conditioning behavior.

#![forbid(unsafe_code)]

use super::manifold::ManifoldError;
use crate::core::{
    collections::{Entry, FastHashMap, SmallBuffer, VertexKeyBuffer, fast_hash_map_with_capacity},
    tds::{SimplexKey, Tds, TdsError, VertexKey},
    vertex::Vertex,
};
use crate::topology::spaces::toroidal::{
    LiftedLinkEdge, LiftedVertexBuffer, LiftedVertexId, lifted_vertex_id,
    normalize_lifted_vertices, periodic_simplex_key,
};
use std::{fmt, ptr};
use thiserror::Error;

type RidgeVertexRefBuffer<'tds, U, const D: usize> = SmallBuffer<&'tds Vertex<U, D>, 8>;

/// Errors returned when parsing raw vertex keys into a ridge candidate.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RidgeCandidateError {
    /// Ridge vertices are only meaningful for dimensions `D >= 2`.
    #[error("ridge candidates require D >= 2, got D={dimension}")]
    UnsupportedDimension {
        /// Requested triangulation dimension.
        dimension: usize,
    },

    /// The supplied vertex count does not match the ridge arity `D - 1`.
    #[error(
        "ridge candidate vertex count mismatch for {dimension}D: expected {expected}, got {actual}"
    )]
    WrongArity {
        /// Requested triangulation dimension.
        dimension: usize,
        /// Expected number of ridge vertices.
        expected: usize,
        /// Actual number of supplied vertices.
        actual: usize,
    },

    /// A ridge cannot contain the same vertex more than once.
    #[error("ridge candidate contains duplicate vertex key {vertex_key:?}")]
    DuplicateVertex {
        /// Duplicate vertex key.
        vertex_key: VertexKey,
    },
}

/// Validated vertex keys for a potential `(D - 2)`-simplex ridge.
///
/// This proof-bearing wrapper encodes the arity and uniqueness invariants for
/// ridge-star queries before they reach topology computation. It stores vertex
/// keys in canonical sorted order so the same candidate has the same identity
/// regardless of input order. It does not prove that the vertices exist in a
/// particular [`Tds`] or that the ridge occurs in any D-simplex; use
/// [`Self::query`] for a borrowed possibly-empty query and [`Self::view`] for a
/// borrowed view that proves a non-empty star.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{
///     ManifoldError, RidgeCandidate, RidgeCandidateError,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Ridge(#[from] RidgeCandidateError),
/// #     #[error(transparent)]
/// #     Manifold(#[from] ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// // In 2D, a ridge is a vertex because it has arity D - 1.
/// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
/// let star = dt.ridge_star_simplices(&ridge)?;
/// let view = dt.ridge_view(&ridge)?;
///
/// assert_eq!(ridge.as_slice(), view.vertex_keys());
/// assert_eq!(star.as_slice(), view.incident_simplices());
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RidgeCandidate<const D: usize> {
    vertices: VertexKeyBuffer,
}

impl<const D: usize> RidgeCandidate<D> {
    /// Parses raw vertex keys into a validated ridge candidate.
    ///
    /// Stored vertex keys are canonicalized into sorted order.
    ///
    /// # Errors
    ///
    /// Returns [`RidgeCandidateError::UnsupportedDimension`] when `D < 2`,
    /// [`RidgeCandidateError::WrongArity`] when the input length is not `D - 1`,
    /// or [`RidgeCandidateError::DuplicateVertex`] when a vertex key is repeated.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<3>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(2))?;
    /// assert_eq!(ridge.as_slice().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_vertices(
        vertices: impl IntoIterator<Item = VertexKey>,
    ) -> Result<Self, RidgeCandidateError> {
        if D < 2 {
            return Err(RidgeCandidateError::UnsupportedDimension { dimension: D });
        }

        let mut vertices: VertexKeyBuffer = vertices.into_iter().collect();
        let expected = D - 1;
        if vertices.len() != expected {
            return Err(RidgeCandidateError::WrongArity {
                dimension: D,
                expected,
                actual: vertices.len(),
            });
        }

        vertices.sort_unstable();
        for duplicate_pair in vertices.windows(2) {
            if duplicate_pair[0] == duplicate_pair[1] {
                return Err(RidgeCandidateError::DuplicateVertex {
                    vertex_key: duplicate_pair[0],
                });
            }
        }

        Ok(Self { vertices })
    }

    /// Returns the validated ridge vertex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<3>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(2))?;
    /// assert_eq!(ridge.as_slice().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn as_slice(&self) -> &[VertexKey] {
        &self.vertices
    }

    /// Iterates over the validated ridge vertex keys.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<3>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(2))?;
    /// assert_eq!(ridge.iter().count(), ridge.as_slice().len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = VertexKey> + '_ {
        self.vertices.iter().copied()
    }

    /// Revalidates this candidate against a live TDS and returns a borrowed query.
    ///
    /// A [`RidgeQuery`] proves only that the candidate's vertices are live in
    /// `tds`. Its incident star may be empty.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if any ridge vertex is stale.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let query = dt.ridge_query(&ridge)?;
    /// assert_eq!(query.vertex_keys(), ridge.as_slice());
    /// # Ok(())
    /// # }
    /// ```
    pub fn query<'tds, U, V>(
        &self,
        tds: &'tds Tds<U, V, D>,
    ) -> Result<RidgeQuery<'tds, U, V, D>, ManifoldError> {
        RidgeQuery::try_new(tds, self.clone())
    }

    /// Revalidates this candidate against a live TDS and returns a borrowed view.
    ///
    /// A [`RidgeView`] proves that the candidate's vertices are live and that
    /// at least one D-simplex is incident to the ridge.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if any ridge vertex is stale or the TDS is
    /// internally inconsistent while resolving the star. Returns
    /// [`ManifoldError::RidgeNotFound`] if the live candidate has an empty star.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let view = dt.ridge_view(&ridge)?;
    /// assert!(!view.incident_simplices().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn view<'tds, U, V>(
        &self,
        tds: &'tds Tds<U, V, D>,
    ) -> Result<RidgeView<'tds, U, V, D>, ManifoldError> {
        RidgeView::try_new(tds, self.clone())
    }
}

/// Borrowed live-TDS query over a ridge candidate.
///
/// `RidgeQuery` is a non-durable topology query. Construction proves the
/// candidate's vertices are live in one borrowed [`Tds`], but it deliberately
/// permits an empty incident star. Use [`RidgeView`] when the API requires a
/// ridge that is known to exist in the triangulation.
#[must_use]
pub struct RidgeQuery<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    ridge_candidate: RidgeCandidate<D>,
    vertices: RidgeVertexRefBuffer<'tds, U, D>,
    star_simplices: SmallBuffer<SimplexKey, 8>,
}

impl<'tds, U, V, const D: usize> RidgeQuery<'tds, U, V, D> {
    /// Creates a borrowed ridge query after validating `ridge_candidate` against `tds`.
    ///
    /// This checks that every candidate vertex is live in `tds`. It does not
    /// require the candidate to occur in any D-simplex; [`Self::incident_simplices`]
    /// and [`Self::links`] return empty buffers for isolated live candidates.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if any ridge vertex is stale or if the
    /// TDS is internally inconsistent while resolving the candidate star.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError, RidgeQuery,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let query = dt.ridge_query(&ridge)?;
    /// assert_eq!(query.vertices().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(
        tds: &'tds Tds<U, V, D>,
        ridge_candidate: RidgeCandidate<D>,
    ) -> Result<Self, ManifoldError> {
        let vertices = resolve_ridge_vertices(tds, &ridge_candidate, "ridge query construction")?;
        let star_simplices = simplex_star_simplices(tds, ridge_candidate.as_slice())?;

        Ok(Self {
            tds,
            ridge_candidate,
            vertices,
            star_simplices,
        })
    }

    /// Returns the validated ridge candidate represented by this query.
    #[inline]
    pub const fn ridge_candidate(&self) -> &RidgeCandidate<D> {
        &self.ridge_candidate
    }

    /// Returns the ridge vertex keys in canonical order.
    #[inline]
    #[must_use]
    pub fn vertex_keys(&self) -> &[VertexKey] {
        self.ridge_candidate.as_slice()
    }

    /// Returns borrowed ridge vertices in canonical key order.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[&'tds Vertex<U, D>] {
        self.vertices.as_slice()
    }

    /// Returns all D-simplices incident to this candidate.
    #[inline]
    #[must_use]
    pub fn incident_simplices(&self) -> &[SimplexKey] {
        self.star_simplices.as_slice()
    }

    /// Returns the lifted links represented by this quotient-space candidate.
    ///
    /// A non-periodic ridge normally has one link. In a periodic triangulation,
    /// one [`RidgeCandidate`] value may correspond to multiple lifted ridge
    /// images, so this returns one [`RidgeLinkView`] per lifted ridge identity.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if the backing TDS is internally
    /// inconsistent while resolving the queried ridge's lifted stars.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let query = dt.ridge_query(&ridge)?;
    /// let links = query.links()?;
    /// assert!(!links.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn links(&self) -> Result<SmallBuffer<RidgeLinkView<'tds, U, V, D>, 8>, ManifoldError> {
        ridge_links_from_star(self.tds, &self.ridge_candidate, &self.star_simplices)
    }
}

impl<U, V, const D: usize> fmt::Debug for RidgeQuery<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RidgeQuery")
            .field("ridge_candidate", &self.ridge_candidate)
            .field("dimension", &D)
            .finish()
    }
}

impl<U, V, const D: usize> Clone for RidgeQuery<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            ridge_candidate: self.ridge_candidate.clone(),
            vertices: self.vertices.clone(),
            star_simplices: self.star_simplices.clone(),
        }
    }
}

impl<U, V, const D: usize> PartialEq for RidgeQuery<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.tds, other.tds) && self.ridge_candidate == other.ridge_candidate
    }
}

impl<U, V, const D: usize> Eq for RidgeQuery<'_, U, V, D> {}

/// Borrowed live-TDS view over an existing ridge.
///
/// `RidgeView` is a non-durable topology view. It borrows one in-memory [`Tds`],
/// owns the validated [`RidgeCandidate`] identity, and caches the non-empty
/// incident D-simplex star proven during construction. Persist stable vertex
/// UUIDs or a full TDS snapshot instead of serializing ridge views.
#[must_use]
pub struct RidgeView<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    ridge_candidate: RidgeCandidate<D>,
    vertices: RidgeVertexRefBuffer<'tds, U, D>,
    star_simplices: SmallBuffer<SimplexKey, 8>,
}

impl<'tds, U, V, const D: usize> RidgeView<'tds, U, V, D> {
    /// Creates a borrowed ridge view after validating `ridge_candidate` against `tds`.
    ///
    /// This checks that every ridge vertex is live in `tds` and that the
    /// candidate occurs in at least one D-simplex.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if any ridge vertex is stale or if the TDS
    /// is internally inconsistent while resolving the star. Returns
    /// [`ManifoldError::RidgeNotFound`] if the live candidate has an empty star.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError, RidgeView,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let view = dt.ridge_view(&ridge)?;
    /// assert_eq!(view.vertices().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(
        tds: &'tds Tds<U, V, D>,
        ridge_candidate: RidgeCandidate<D>,
    ) -> Result<Self, ManifoldError> {
        let query = RidgeQuery::try_new(tds, ridge_candidate)?;
        if query.star_simplices.is_empty() {
            return Err(ManifoldError::RidgeNotFound {
                ridge_vertices: query.ridge_candidate.as_slice().iter().copied().collect(),
            });
        }

        Ok(Self {
            tds: query.tds,
            ridge_candidate: query.ridge_candidate,
            vertices: query.vertices,
            star_simplices: query.star_simplices,
        })
    }

    /// Returns the validated ridge candidate represented by this view.
    #[inline]
    pub const fn ridge_candidate(&self) -> &RidgeCandidate<D> {
        &self.ridge_candidate
    }

    /// Returns the ridge vertex keys in canonical order.
    #[inline]
    #[must_use]
    pub fn vertex_keys(&self) -> &[VertexKey] {
        self.ridge_candidate.as_slice()
    }

    /// Returns borrowed ridge vertices in canonical key order.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[&'tds Vertex<U, D>] {
        self.vertices.as_slice()
    }

    /// Returns all D-simplices incident to this ridge.
    #[inline]
    #[must_use]
    pub fn incident_simplices(&self) -> &[SimplexKey] {
        self.star_simplices.as_slice()
    }

    /// Returns the lifted links represented by this quotient-space ridge.
    ///
    /// A non-periodic ridge normally has one link. In a periodic triangulation,
    /// one [`RidgeCandidate`] value may correspond to multiple lifted ridge
    /// images, so this returns one [`RidgeLinkView`] per lifted ridge identity.
    ///
    /// # Errors
    ///
    /// Returns [`ManifoldError::Tds`] if the backing TDS is internally
    /// inconsistent while resolving the queried ridge's lifted stars.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::*;
    /// use delaunay::prelude::topology::validation::{
    ///     ManifoldError, RidgeCandidate, RidgeCandidateError,
    /// };
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Ridge(#[from] RidgeCandidateError),
    /// #     #[error(transparent)]
    /// #     Manifold(#[from] ManifoldError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    ///
    /// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
    /// let view = dt.ridge_view(&ridge)?;
    /// let links = view.links()?;
    /// assert_eq!(links[0].quotient_ridge_candidate(), view.ridge_candidate());
    /// # Ok(())
    /// # }
    /// ```
    pub fn links(&self) -> Result<SmallBuffer<RidgeLinkView<'tds, U, V, D>, 8>, ManifoldError> {
        ridge_links_from_star(self.tds, &self.ridge_candidate, &self.star_simplices)
    }
}

impl<U, V, const D: usize> fmt::Debug for RidgeView<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RidgeView")
            .field("ridge_candidate", &self.ridge_candidate)
            .field("star_simplices", &self.star_simplices)
            .field("dimension", &D)
            .finish()
    }
}

impl<U, V, const D: usize> Clone for RidgeView<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            ridge_candidate: self.ridge_candidate.clone(),
            vertices: self.vertices.clone(),
            star_simplices: self.star_simplices.clone(),
        }
    }
}

impl<U, V, const D: usize> PartialEq for RidgeView<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.tds, other.tds)
            && self.ridge_candidate == other.ridge_candidate
            && self.star_simplices == other.star_simplices
    }
}

impl<U, V, const D: usize> Eq for RidgeView<'_, U, V, D> {}

/// Borrowed view of one lifted ridge link.
///
/// This view is lifetime-bound to the [`Tds`] that produced it, while owning the
/// small derived star for one lifted ridge image. Holding the borrow prevents
/// topology mutation while callers inspect lifted link edges.
///
/// Callers obtain this view from [`RidgeQuery::links`] or [`RidgeView::links`].
/// Its lifted vertices and
/// [`LiftedLinkEdge`] values are runtime topology observations, not durable
/// storage identities.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{
///     ManifoldError, RidgeCandidate, RidgeCandidateError,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Ridge(#[from] RidgeCandidateError),
/// #     #[error(transparent)]
/// #     Manifold(#[from] ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(1))?;
/// let view = dt.ridge_view(&ridge)?;
/// let links = view.links()?;
/// let link = &links[0];
///
/// assert_eq!(link.incident_simplices(), view.incident_simplices());
/// assert_eq!(link.quotient_ridge_candidate(), view.ridge_candidate());
/// assert_eq!(link.lifted_ridge_vertices().len(), ridge.as_slice().len());
/// assert!(!link.edges().is_empty());
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct RidgeLinkView<'tds, U, V, const D: usize> {
    tds: &'tds Tds<U, V, D>,
    quotient_ridge_candidate: RidgeCandidate<D>,
    lifted_ridge_vertices: LiftedVertexBuffer,
    star_simplices: SmallBuffer<SimplexKey, 8>,
    link_edges: SmallBuffer<LiftedLinkEdge, 8>,
}

impl<U, V, const D: usize> RidgeLinkView<'_, U, V, D> {
    /// Returns the quotient-space ridge candidate that produced this link.
    #[inline]
    pub const fn quotient_ridge_candidate(&self) -> &RidgeCandidate<D> {
        &self.quotient_ridge_candidate
    }

    /// Returns the lifted ridge vertices for this particular link image.
    #[inline]
    pub fn lifted_ridge_vertices(&self) -> &[LiftedVertexId] {
        &self.lifted_ridge_vertices
    }

    /// Returns the D-simplices incident to this lifted ridge image.
    #[inline]
    #[must_use]
    pub fn incident_simplices(&self) -> &[SimplexKey] {
        &self.star_simplices
    }

    /// Returns lifted edges in this ridge's 1-dimensional link.
    #[inline]
    pub fn edges(&self) -> &[LiftedLinkEdge] {
        &self.link_edges
    }
}

impl<U, V, const D: usize> fmt::Debug for RidgeLinkView<'_, U, V, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RidgeLinkView")
            .field("quotient_ridge_candidate", &self.quotient_ridge_candidate)
            .field("lifted_ridge_vertices", &self.lifted_ridge_vertices)
            .field("star_simplices", &self.star_simplices)
            .field("link_edges", &self.link_edges)
            .field("dimension", &D)
            .finish()
    }
}

impl<U, V, const D: usize> Clone for RidgeLinkView<'_, U, V, D> {
    fn clone(&self) -> Self {
        Self {
            tds: self.tds,
            quotient_ridge_candidate: self.quotient_ridge_candidate.clone(),
            lifted_ridge_vertices: self.lifted_ridge_vertices.clone(),
            star_simplices: self.star_simplices.clone(),
            link_edges: self.link_edges.clone(),
        }
    }
}

impl<U, V, const D: usize> PartialEq for RidgeLinkView<'_, U, V, D> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self.tds, other.tds)
            && self.quotient_ridge_candidate == other.quotient_ridge_candidate
            && self.lifted_ridge_vertices == other.lifted_ridge_vertices
            && self.star_simplices == other.star_simplices
            && self.link_edges == other.link_edges
    }
}

impl<U, V, const D: usize> Eq for RidgeLinkView<'_, U, V, D> {}

/// Resolves a parsed ridge candidate to borrowed live vertices.
///
/// This is the boundary that turns detached [`RidgeCandidate`] values into
/// live-TDS [`RidgeQuery`] and [`RidgeView`] values. After this succeeds, those
/// views can return borrowed ridge vertices infallibly for the lifetime of the
/// TDS borrow.
fn resolve_ridge_vertices<'tds, U, V, const D: usize>(
    tds: &'tds Tds<U, V, D>,
    ridge_candidate: &RidgeCandidate<D>,
    context: &str,
) -> Result<RidgeVertexRefBuffer<'tds, U, D>, ManifoldError> {
    let mut vertices = RidgeVertexRefBuffer::with_capacity(ridge_candidate.as_slice().len());
    for &vertex_key in ridge_candidate.as_slice() {
        let vertex = tds
            .vertex(vertex_key)
            .ok_or_else(|| TdsError::VertexNotFound {
                vertex_key,
                context: context.to_string(),
            })?;
        vertices.push(vertex);
    }

    Ok(vertices)
}

/// Groups a quotient ridge star into one view per lifted ridge image.
///
/// Periodic triangulations can represent several covering-space ridge images
/// with one quotient [`RidgeCandidate`]. This helper preserves those images so
/// [`RidgeQuery::links`] and [`RidgeView::links`] do not collapse distinct
/// toroidal link components.
fn ridge_links_from_star<'tds, U, V, const D: usize>(
    tds: &'tds Tds<U, V, D>,
    ridge_candidate: &RidgeCandidate<D>,
    incident_simplices: &[SimplexKey],
) -> Result<SmallBuffer<RidgeLinkView<'tds, U, V, D>, 8>, ManifoldError> {
    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(incident_simplices.len().max(1));

    for &simplex_key in incident_simplices {
        let lifted_vertex_images =
            simplex_lifted_ridge_vertex_images(tds, simplex_key, ridge_candidate)?;
        for lifted_vertices in lifted_vertex_images {
            let ridge_key = periodic_simplex_key(&lifted_vertices);
            match ridge_to_star.entry(ridge_key) {
                Entry::Occupied(_) => {}
                Entry::Vacant(entry) => {
                    let star_simplices = periodic_aware_ridge_star(
                        tds,
                        ridge_key,
                        &lifted_vertices,
                        ridge_candidate.as_slice(),
                    )?;
                    entry.insert(RidgeStar {
                        ridge_vertices: lifted_vertices,
                        star_simplices,
                    });
                }
            }
        }
    }

    let mut links: SmallBuffer<RidgeLinkView<'tds, U, V, D>, 8> = SmallBuffer::new();
    for star in ridge_to_star.into_values() {
        let link_edges =
            lifted_link_edges_from_star(tds, &star.ridge_vertices, &star.star_simplices)?;
        links.push(RidgeLinkView {
            tds,
            quotient_ridge_candidate: ridge_candidate.clone(),
            lifted_ridge_vertices: star.ridge_vertices,
            star_simplices: star.star_simplices,
            link_edges,
        });
    }

    links.sort_unstable_by(|a, b| {
        a.lifted_ridge_vertices
            .as_slice()
            .cmp(b.lifted_ridge_vertices.as_slice())
    });
    Ok(links)
}

/// Parses a lifted ridge star into the lifted link edges stored by
/// [`RidgeLinkView`].
fn lifted_link_edges_from_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_vertices: &[LiftedVertexId],
    star_simplices: &[SimplexKey],
) -> Result<SmallBuffer<LiftedLinkEdge, 8>, ManifoldError> {
    ridge_link_edges_from_star(tds, ridge_vertices, star_simplices).map(|edges| {
        edges
            .into_iter()
            .map(|(a, b)| LiftedLinkEdge::from_unordered_endpoints(&a, &b))
            .collect()
    })
}

/// Computes the star of a simplex (a set of vertices) as the set of incident D-simplices.
///
/// This helper does **not** call `tds.is_valid()`; it performs lightweight checks and
/// returns [`ManifoldError::Tds`] if the underlying TDS is internally inconsistent.
pub(crate) fn simplex_star_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_vertices: &[VertexKey],
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    if simplex_vertices.is_empty() {
        return Err(TdsError::DimensionMismatch {
            expected: 1,
            actual: 0,
            context: "simplex_star_simplices requires at least one vertex".to_string(),
        }
        .into());
    }

    for &vk in simplex_vertices {
        if !tds.contains_vertex_key(vk) {
            return Err(TdsError::VertexNotFound {
                vertex_key: vk,
                context: "simplex star computation".to_string(),
            }
            .into());
        }
    }

    let candidates = tds.simplex_keys_containing_vertex(simplex_vertices[0]);
    let mut star_simplices: SmallBuffer<SimplexKey, 8> = SmallBuffer::new();

    for simplex_key in candidates {
        let candidate_vertices = tds.simplex_vertices(simplex_key)?;
        if simplex_vertices
            .iter()
            .all(|&sv| candidate_vertices.contains(&sv))
        {
            star_simplices.push(simplex_key);
        }
    }

    Ok(star_simplices)
}

/// Computes the star of a ridge candidate as the set of incident D-simplices.
///
/// Prefer [`RidgeCandidate::query`] or [`RidgeCandidate::view`] when a caller
/// also needs borrowed ridge vertices. This helper is the lightweight public
/// entry point for star enumeration.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] when any ridge vertex is missing from the
/// [`Tds`] or a candidate star simplex cannot resolve its vertex keys.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::*;
/// use delaunay::prelude::topology::validation::{
///     ManifoldError, RidgeCandidate, RidgeCandidateError, ridge_star_simplices,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Ridge(#[from] RidgeCandidateError),
/// #     #[error(transparent)]
/// #     Manifold(#[from] ManifoldError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// // In 3D, a ridge is an edge because it has arity D - 1.
/// let ridge = RidgeCandidate::<3>::try_from_vertices(dt.vertices().map(|(key, _)| key).take(2))?;
/// let star = dt.ridge_star_simplices(&ridge)?;
/// assert!(!star.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn ridge_star_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_candidate: &RidgeCandidate<D>,
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    simplex_star_simplices(tds, ridge_candidate.as_slice())
}

/// Extracts every lifted image of a ridge candidate in one simplex frame.
///
/// The returned images are normalized so periodic ridge identity is stable
/// across adjacent quotient simplices. Public ridge-link APIs rely on this to
/// keep distinct toroidal covering-space images separate.
///
/// If a quotient vertex occurs through multiple lifted simplex slots, this
/// helper returns every normalized image combination instead of collapsing to
/// the first matching slot.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if the simplex vertex count or periodic offset
/// count is inconsistent, or if the simplex does not contain every quotient
/// vertex in the [`RidgeCandidate`].
fn simplex_lifted_ridge_vertex_images<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    ridge_vertices: &RidgeCandidate<D>,
) -> Result<SmallBuffer<LiftedVertexBuffer, 8>, ManifoldError> {
    let simplex_vertices = tds.simplex_vertices(simplex_key)?;
    if simplex_vertices.len() != D + 1 {
        return Err(TdsError::DimensionMismatch {
            expected: D + 1,
            actual: simplex_vertices.len(),
            context: format!("simplex {simplex_key:?} vertex count for {D}D (ridge view)"),
        }
        .into());
    }

    let offsets = tds
        .simplex(simplex_key)
        .and_then(|simplex| simplex.periodic_vertex_offsets());
    if let Some(simplex_offsets) = offsets
        && simplex_offsets.len() != simplex_vertices.len()
    {
        return Err(TdsError::DimensionMismatch {
            expected: simplex_vertices.len(),
            actual: simplex_offsets.len(),
            context: format!("periodic offset count for {D}D ridge view simplex {simplex_key:?}"),
        }
        .into());
    }

    let mut lifted_vertex_images: SmallBuffer<LiftedVertexBuffer, 8> = SmallBuffer::new();
    lifted_vertex_images.push(LiftedVertexBuffer::new());

    for &vertex_key in ridge_vertices.as_slice() {
        let mut lifted_occurrences = LiftedVertexBuffer::new();
        for (vertex_index, &candidate) in simplex_vertices.iter().enumerate() {
            if candidate != vertex_key {
                continue;
            }
            let lifted = offsets.map_or_else(
                || LiftedVertexId::base(vertex_key),
                |simplex_offsets| {
                    lifted_vertex_id(
                        vertex_key,
                        simplex_offsets[vertex_index].iter().copied().map(i16::from),
                    )
                },
            );
            lifted_occurrences.push(lifted);
        }

        if lifted_occurrences.is_empty() {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "ridge view simplex {simplex_key:?} does not contain ridge vertex {vertex_key:?}"
                ),
            }
            .into());
        }

        let mut next_images: SmallBuffer<LiftedVertexBuffer, 8> = SmallBuffer::with_capacity(
            lifted_vertex_images
                .len()
                .saturating_mul(lifted_occurrences.len()),
        );
        for image in &lifted_vertex_images {
            for lifted in &lifted_occurrences {
                let mut next_image = image.clone();
                next_image.push(lifted.clone());
                next_images.push(next_image);
            }
        }
        lifted_vertex_images = next_images;
    }

    let mut normalized_images: SmallBuffer<LiftedVertexBuffer, 8> =
        SmallBuffer::with_capacity(lifted_vertex_images.len());
    for lifted_vertices in lifted_vertex_images {
        if lifted_vertices.len() != D.saturating_sub(1) {
            return Err(TdsError::DimensionMismatch {
                expected: D.saturating_sub(1),
                actual: lifted_vertices.len(),
                context: format!(
                    "ridge vertex count for {D}D (ridge view simplex {simplex_key:?})"
                ),
            }
            .into());
        }

        let normalized = normalize_lifted_vertices(&lifted_vertices);
        if normalized_images
            .iter()
            .all(|existing| existing.as_slice() != normalized.as_slice())
        {
            normalized_images.push(normalized);
        }
    }

    Ok(normalized_images)
}

/// Builds lifted link edges for the star of one lifted ridge image.
///
/// This implements the construction-time parse boundary for
/// [`RidgeLinkView`]: every lifted occurrence of every star simplex must
/// contribute exactly two complementary link vertices, and those vertices must
/// form a valid lifted edge.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if the ridge arity is wrong, a star simplex
/// does not contain the requested lifted ridge image, a complementary link does
/// not contain exactly two vertices, or a lifted link edge would be a self-loop.
pub(crate) fn ridge_link_edges_from_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_vertices: &[LiftedVertexId],
    star_simplices: &[SimplexKey],
) -> Result<SmallBuffer<(LiftedVertexId, LiftedVertexId), 8>, ManifoldError> {
    if D < 2 {
        return Ok(SmallBuffer::new());
    }

    let expected_ridge_vertices = D.saturating_sub(1);
    if ridge_vertices.len() != expected_ridge_vertices {
        return Err(TdsError::DimensionMismatch {
            expected: expected_ridge_vertices,
            actual: ridge_vertices.len(),
            context: format!("ridge vertex count for {D}D (link edges)"),
        }
        .into());
    }

    let mut link_edges: SmallBuffer<(LiftedVertexId, LiftedVertexId), 8> =
        SmallBuffer::with_capacity(star_simplices.len());
    let mut link_vertices: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(2);

    for &simplex_key in star_simplices {
        let simplex_vertex_images =
            normalized_simplex_vertices_for_lifted_target(tds, simplex_key, ridge_vertices)?;
        if simplex_vertex_images.is_empty() {
            return Err(TdsError::InconsistentDataStructure {
                message: format!(
                    "ridge star simplex {simplex_key:?} does not contain normalized ridge vertices \
                     {ridge_vertices:?}"
                ),
            }
            .into());
        }

        for simplex_vertices in simplex_vertex_images {
            link_vertices.clear();
            for lifted in simplex_vertices {
                if !ridge_vertices.contains(&lifted) {
                    link_vertices.push(lifted);
                }
            }

            if link_vertices.len() != 2 {
                return Err(TdsError::DimensionMismatch {
                    expected: 2,
                    actual: link_vertices.len(),
                    context: format!(
                        "ridge link vertex count for {D}D (simplex_key={simplex_key:?})"
                    ),
                }
                .into());
            }

            if link_vertices[0] == link_vertices[1] {
                return Err(TdsError::InconsistentDataStructure {
                    message: format!(
                        "Ridge link edge is a self-loop: link vertex {vk:?} repeated (simplex_key={simplex_key:?})",
                        vk = &link_vertices[0],
                    ),
                }
                .into());
            }

            link_edges.push((link_vertices[0].clone(), link_vertices[1].clone()));
        }
    }

    Ok(link_edges)
}

/// Lifted ridge identity paired with the simplices incident to that image.
///
/// Validators and ridge views use this derived value to avoid collapsing
/// distinct toroidal covering-space stars that share the same quotient
/// [`VertexKey`] set.
#[derive(Clone, Debug)]
pub(crate) struct RidgeStar {
    pub(crate) ridge_vertices: LiftedVertexBuffer,
    pub(crate) star_simplices: SmallBuffer<SimplexKey, 8>,
}

/// Builds a complete ridge-to-star incidence map for one TDS.
///
/// This is the shared topology-validation path for ridge multiplicity and link
/// checks. It visits every simplex once and enumerates its ridges, preserving
/// lifted toroidal identity in the map key.
pub(crate) fn build_ridge_star_map<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<FastHashMap<u64, RidgeStar>, ManifoldError> {
    let simplex_count = tds.number_of_simplices();
    if D < 2 || simplex_count == 0 {
        return Ok(FastHashMap::default());
    }

    let ridges_per_simplex = (D + 1).saturating_mul(D) / 2;
    let estimated_unique_ridges = simplex_count
        .saturating_mul(ridges_per_simplex)
        .saturating_div(2)
        .max(1);

    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(estimated_unique_ridges);
    let mut ridge_vertices: LiftedVertexBuffer =
        LiftedVertexBuffer::with_capacity(D.saturating_sub(1));

    for (simplex_key, simplex) in tds.simplices() {
        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        let offsets = simplex.periodic_vertex_offsets();

        if simplex_vertices.len() != D + 1 {
            return Err(TdsError::DimensionMismatch {
                expected: D + 1,
                actual: simplex_vertices.len(),
                context: format!("simplex {simplex_key:?} vertex count for {D}D"),
            }
            .into());
        }
        if let Some(simplex_offsets) = offsets
            && simplex_offsets.len() != simplex_vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: simplex_vertices.len(),
                actual: simplex_offsets.len(),
                context: format!(
                    "periodic offset count for {D}D simplex {simplex_key:?} (ridge map)"
                ),
            }
            .into());
        }

        for omit_a in 0..simplex_vertices.len() {
            for omit_b in (omit_a + 1)..simplex_vertices.len() {
                ridge_vertices.clear();
                for (i, &vk) in simplex_vertices.iter().enumerate() {
                    if i == omit_a || i == omit_b {
                        continue;
                    }
                    let lifted = offsets.map_or_else(
                        || LiftedVertexId::base(vk),
                        |offs| lifted_vertex_id(vk, offs[i].iter().copied().map(i16::from)),
                    );
                    ridge_vertices.push(lifted);
                }

                if ridge_vertices.len() != D.saturating_sub(1) {
                    return Err(TdsError::DimensionMismatch {
                        expected: D.saturating_sub(1),
                        actual: ridge_vertices.len(),
                        context: format!("ridge vertex count for {D}D (simplex_key={simplex_key:?}, omit_a={omit_a}, omit_b={omit_b})"),
                    }
                    .into());
                }

                let normalized_ridge_vertices = normalize_lifted_vertices(&ridge_vertices);
                let ridge_key = periodic_simplex_key(&normalized_ridge_vertices);
                let star = ridge_to_star.entry(ridge_key).or_insert_with(|| RidgeStar {
                    ridge_vertices: normalized_ridge_vertices,
                    star_simplices: SmallBuffer::new(),
                });
                star.star_simplices.push(simplex_key);
            }
        }
    }

    Ok(ridge_to_star)
}

/// Builds ridge stars whose seed ridges appear in the supplied simplices.
///
/// Repair and localized validation use this to avoid a full incidence rebuild.
/// Each discovered ridge is expanded through [`periodic_aware_ridge_star`] so
/// the returned stars are complete for the corresponding lifted ridge image.
pub(crate) fn build_ridge_star_map_for_simplices<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplices: impl IntoIterator<Item = SimplexKey>,
) -> Result<FastHashMap<u64, RidgeStar>, ManifoldError> {
    if D < 2 {
        return Ok(FastHashMap::default());
    }

    let simplices = simplices.into_iter();
    let (lower_bound, upper_bound) = simplices.size_hint();
    let estimated_simplex_count = upper_bound.unwrap_or(lower_bound);
    let ridges_per_simplex = (D + 1).saturating_mul(D) / 2;
    let estimated_unique_ridges = estimated_simplex_count
        .saturating_mul(ridges_per_simplex)
        .max(1);

    let mut ridge_to_vertices: FastHashMap<u64, (LiftedVertexBuffer, VertexKeyBuffer)> =
        fast_hash_map_with_capacity(estimated_unique_ridges);
    let mut ridge_vertices_bare: VertexKeyBuffer =
        VertexKeyBuffer::with_capacity(D.saturating_sub(1));
    let mut ridge_vertices_lifted: LiftedVertexBuffer =
        LiftedVertexBuffer::with_capacity(D.saturating_sub(1));

    for simplex_key in simplices {
        if !tds.contains_simplex(simplex_key) {
            continue;
        }

        let simplex_vertices = tds.simplex_vertices(simplex_key)?;
        let offsets = tds
            .simplex(simplex_key)
            .and_then(|c| c.periodic_vertex_offsets());

        if simplex_vertices.len() != D + 1 {
            return Err(TdsError::DimensionMismatch {
                expected: D + 1,
                actual: simplex_vertices.len(),
                context: format!("simplex {simplex_key:?} vertex count for {D}D (local ridge map)"),
            }
            .into());
        }
        if let Some(simplex_offsets) = offsets
            && simplex_offsets.len() != simplex_vertices.len()
        {
            return Err(TdsError::DimensionMismatch {
                expected: simplex_vertices.len(),
                actual: simplex_offsets.len(),
                context: format!(
                    "periodic offset count for {D}D simplex {simplex_key:?} (local ridge map)"
                ),
            }
            .into());
        }

        for omit_a in 0..simplex_vertices.len() {
            for omit_b in (omit_a + 1)..simplex_vertices.len() {
                ridge_vertices_bare.clear();
                ridge_vertices_lifted.clear();
                for (i, &vk) in simplex_vertices.iter().enumerate() {
                    if i == omit_a || i == omit_b {
                        continue;
                    }
                    ridge_vertices_bare.push(vk);
                    let lifted = offsets.map_or_else(
                        || LiftedVertexId::base(vk),
                        |offs| lifted_vertex_id(vk, offs[i].iter().copied().map(i16::from)),
                    );
                    ridge_vertices_lifted.push(lifted);
                }

                if ridge_vertices_bare.len() != D.saturating_sub(1) {
                    return Err(TdsError::DimensionMismatch {
                        expected: D.saturating_sub(1),
                        actual: ridge_vertices_bare.len(),
                        context: format!("ridge vertex count for {D}D (simplex_key={simplex_key:?}, omit_a={omit_a}, omit_b={omit_b})"),
                    }
                    .into());
                }

                let normalized_ridge_vertices = normalize_lifted_vertices(&ridge_vertices_lifted);
                let ridge_key = periodic_simplex_key(&normalized_ridge_vertices);
                ridge_to_vertices
                    .entry(ridge_key)
                    .or_insert_with(|| (normalized_ridge_vertices, ridge_vertices_bare.clone()));
            }
        }
    }

    let mut ridge_to_star: FastHashMap<u64, RidgeStar> =
        fast_hash_map_with_capacity(ridge_to_vertices.len().max(1));

    for (ridge_key, (lifted_vertices, bare_vertices)) in ridge_to_vertices {
        let star_simplices =
            periodic_aware_ridge_star(tds, ridge_key, &lifted_vertices, &bare_vertices)?;
        ridge_to_star.insert(
            ridge_key,
            RidgeStar {
                ridge_vertices: lifted_vertices,
                star_simplices,
            },
        );
    }

    Ok(ridge_to_star)
}

/// Expresses every matching simplex image in the frame of a lifted target ridge.
///
/// Returns an empty buffer when the simplex is not incident to the target ridge
/// image. Otherwise, each returned buffer contains the simplex vertices
/// normalized against one actual matching lifted anchor occurrence. Public
/// ridge-link construction uses this to preserve periodic self-identification
/// links instead of anchoring every match to the first quotient key.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if the simplex has malformed periodic offset
/// metadata.
pub(crate) fn normalized_simplex_vertices_for_lifted_target<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    simplex_key: SimplexKey,
    target_vertices: &[LiftedVertexId],
) -> Result<SmallBuffer<LiftedVertexBuffer, 8>, ManifoldError> {
    let simplex_vertices = tds.simplex_vertices(simplex_key)?;
    let offsets = tds
        .simplex(simplex_key)
        .and_then(|simplex| simplex.periodic_vertex_offsets());
    let mut matching_images: SmallBuffer<LiftedVertexBuffer, 8> = SmallBuffer::new();

    let Some(offsets) = offsets else {
        let vertices: LiftedVertexBuffer = simplex_vertices
            .iter()
            .copied()
            .map(LiftedVertexId::base)
            .collect();
        if target_vertices
            .iter()
            .all(|target| vertices.contains(target))
        {
            matching_images.push(vertices);
        }
        return Ok(matching_images);
    };
    if offsets.len() != simplex_vertices.len() {
        return Err(TdsError::DimensionMismatch {
            expected: simplex_vertices.len(),
            actual: offsets.len(),
            context: format!(
                "periodic offset count for {D}D simplex {simplex_key:?} \
                 (lifted target normalization)"
            ),
        }
        .into());
    }

    let Some(anchor) = target_vertices.first() else {
        matching_images.push(LiftedVertexBuffer::new());
        return Ok(matching_images);
    };

    for (anchor_index, &anchor_key) in simplex_vertices.iter().enumerate() {
        if anchor_key != anchor.vertex_key {
            continue;
        }

        let anchor_offset = offsets[anchor_index];
        let mut normalized = LiftedVertexBuffer::with_capacity(simplex_vertices.len());
        for (idx, &vertex_key) in simplex_vertices.iter().enumerate() {
            let mut relative_offset: SmallBuffer<i16, 8> = SmallBuffer::with_capacity(D);
            for axis in 0..D {
                let target_anchor_component = anchor.offset().get(axis).copied().unwrap_or(0);
                relative_offset.push(
                    i16::from(offsets[idx][axis]) - i16::from(anchor_offset[axis])
                        + target_anchor_component,
                );
            }
            normalized.push(lifted_vertex_id(vertex_key, relative_offset));
        }
        if target_vertices
            .iter()
            .all(|target| normalized.contains(target))
            && matching_images
                .iter()
                .all(|existing| existing.as_slice() != normalized.as_slice())
        {
            matching_images.push(normalized);
        }
    }

    Ok(matching_images)
}

/// Filters a quotient ridge star down to one lifted toroidal ridge image.
///
/// The bare vertex star supplies candidate simplices, while `ridge_key` and
/// `lifted_vertices` identify the covering-space image that public
/// [`RidgeLinkView`] values must preserve.
///
/// # Errors
///
/// Returns [`ManifoldError::Tds`] if the quotient star cannot be queried, if
/// candidate simplices have malformed periodic offset metadata, or if no
/// candidate simplex represents the requested lifted ridge image.
pub(crate) fn periodic_aware_ridge_star<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    ridge_key: u64,
    lifted_vertices: &[LiftedVertexId],
    bare_vertices: &[VertexKey],
) -> Result<SmallBuffer<SimplexKey, 8>, ManifoldError> {
    let all_star_simplices = simplex_star_simplices(tds, bare_vertices)?;
    let mut star_simplices: SmallBuffer<SimplexKey, 8> =
        SmallBuffer::with_capacity(all_star_simplices.len());

    for &simplex_key in &all_star_simplices {
        if !normalized_simplex_vertices_for_lifted_target(tds, simplex_key, lifted_vertices)?
            .is_empty()
        {
            star_simplices.push(simplex_key);
        }
    }

    if star_simplices.is_empty() {
        return Err(TdsError::InconsistentDataStructure {
            message: format!(
                "periodic offset filtering produced empty star for ridge \
                 {ridge_key:016x}: {count} candidate simplices were all excluded \
                 (lifted ridge vertices: {lifted:?})",
                count = all_star_simplices.len(),
                lifted = lifted_vertices,
            ),
        }
        .into());
    }

    Ok(star_simplices)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::{
        collections::{FastHashSet, SimplexKeySet},
        facet::facet_key_from_vertices,
        simplex::Simplex,
    };
    use crate::vertex;
    use slotmap::{Key, KeyData};
    use std::iter;

    type DuplicateLiftedAnchorFixture3d =
        (Tds<(), (), 3>, SimplexKey, VertexKey, VertexKey, VertexKey);

    fn test_vertex<const D: usize>(coords: [f64; D]) -> Vertex<(), D> {
        vertex!(coords).unwrap()
    }

    fn simplex(vertices: &[VertexKey]) -> LiftedVertexBuffer {
        let mut simplex: LiftedVertexBuffer = LiftedVertexBuffer::with_capacity(vertices.len());
        simplex.extend(vertices.iter().copied().map(LiftedVertexId::base));
        simplex
    }

    fn build_duplicate_lifted_anchor_fixture_3d() -> DuplicateLiftedAnchorFixture3d {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
            simplex.push_vertex_key(v2);
            simplex
                .set_periodic_vertex_offsets(vec![[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
                .unwrap();
        }

        (tds, simplex_key, v0, v1, v2)
    }

    fn build_two_tetrahedra_sharing_facet_tds_3d()
    -> (Tds<(), (), 3>, [VertexKey; 5], [SimplexKey; 2]) {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();

        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, -1.0]))
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c2 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v4], None).unwrap(),
            )
            .unwrap();

        (tds, [v0, v1, v2, v3, v4], [c1, c2])
    }

    fn build_wedge_two_spheres_share_vertex_tds_2d()
    -> (Tds<(), (), 2>, VertexKey, SimplexKey, SimplexKey) {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();

        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 1.0]))
            .unwrap();

        let c012 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let _c013 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let _c023 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 10.0]))
            .unwrap();
        let v5 = tds
            .insert_vertex_with_mapping(test_vertex([11.0, 10.0]))
            .unwrap();
        let v6 = tds
            .insert_vertex_with_mapping(test_vertex([10.0, 11.0]))
            .unwrap();

        let _c045 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v5], None).unwrap(),
            )
            .unwrap();
        let _c046 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v4, v6], None).unwrap(),
            )
            .unwrap();
        let _c056 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v5, v6], None).unwrap(),
            )
            .unwrap();
        let _c456 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v4, v5, v6], None).unwrap(),
            )
            .unwrap();

        (tds, v0, c012, c123)
    }

    #[test]
    fn test_simplex_star_simplices_errors_on_empty_simplex() {
        let tds: Tds<(), (), 2> = Tds::empty();

        match simplex_star_simplices(&tds, &[]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 1,
                actual: 0,
                ..
            })) => {}
            other => panic!("Expected DimensionMismatch for empty simplex, got {other:?}"),
        }
    }

    #[test]
    fn test_simplex_star_simplices_returns_empty_for_isolated_vertex() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();

        let star = simplex_star_simplices(&tds, &[v0]).unwrap();
        assert!(star.is_empty());

        let ridge_candidate = RidgeCandidate::<2>::try_from_vertices([v0]).unwrap();
        let ridge_query = ridge_candidate.query(&tds).unwrap();
        assert!(ridge_query.incident_simplices().is_empty());
        assert!(ridge_query.links().unwrap().is_empty());
        match ridge_candidate.view(&tds) {
            Err(ManifoldError::RidgeNotFound { ridge_vertices }) => {
                assert_eq!(ridge_vertices.as_slice(), &[v0]);
            }
            other => panic!("Expected RidgeNotFound for isolated live ridge, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_candidate_rejects_d_lt_2() {
        match RidgeCandidate::<1>::try_from_vertices([VertexKey::from(KeyData::from_ffi(0))]) {
            Err(RidgeCandidateError::UnsupportedDimension { dimension }) => {
                assert_eq!(dimension, 1);
            }
            other => panic!("Expected UnsupportedDimension for D<2, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_noop_for_d_lt_2() {
        let tds: Tds<(), (), 1> = Tds::empty();

        let edges = ridge_link_edges_from_star(&tds, &[], &[]).unwrap();
        assert!(edges.is_empty());
    }

    #[test]
    fn test_ridge_candidate_rejects_too_few_vertices_in_3d() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));

        match RidgeCandidate::<3>::try_from_vertices([v0]) {
            Err(RidgeCandidateError::WrongArity {
                dimension: 3,
                expected: 2,
                actual: 1,
            }) => {}
            other => panic!("Expected WrongArity(2, 1) for wrong ridge size, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_errors_on_wrong_vertex_count_in_3d() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        match ridge_link_edges_from_star(&tds, &simplex(&[v0]), &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 2,
                actual: 1,
                ..
            })) => {}
            other => panic!("Expected DimensionMismatch(2, 1) for wrong ridge size, got {other:?}"),
        }
    }

    #[test]
    fn test_normalized_simplex_vertices_for_lifted_target_empty_for_missing_vertex() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let missing_from_simplex = tds
            .insert_vertex_with_mapping(test_vertex([2.0, 2.0, 2.0]))
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let target: LiftedVertexBuffer = [
            LiftedVertexId::base(v0),
            LiftedVertexId::base(missing_from_simplex),
        ]
        .into_iter()
        .collect();

        assert!(
            normalized_simplex_vertices_for_lifted_target(&tds, simplex_key, &target)
                .unwrap()
                .is_empty()
        );

        tds.simplex_mut(simplex_key)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0, 0]; 4])
            .unwrap();

        assert!(
            normalized_simplex_vertices_for_lifted_target(&tds, simplex_key, &target)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_normalized_simplex_vertices_for_lifted_target_preserves_target_anchor_offset() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        let mut simplex = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex
            .set_periodic_vertex_offsets(vec![[1, 0], [2, 0], [1, 1]])
            .unwrap();
        let simplex_key = tds.insert_simplex_with_mapping(simplex).unwrap();

        let target_anchor = lifted_vertex_id(v0, [3_i16, 0]);
        let target: LiftedVertexBuffer = iter::once(target_anchor.clone()).collect();

        let images =
            normalized_simplex_vertices_for_lifted_target(&tds, simplex_key, &target).unwrap();

        assert_eq!(images.len(), 1);
        assert!(images[0].contains(&target_anchor));
        assert!(images[0].contains(&lifted_vertex_id(v1, [4_i16, 0])));
        assert!(images[0].contains(&lifted_vertex_id(v2, [3_i16, 1])));
    }

    #[test]
    fn test_simplex_lifted_ridge_vertex_images_enumerates_duplicate_lifted_slots() {
        let (tds, simplex_key, v0, v1, _v2) = build_duplicate_lifted_anchor_fixture_3d();
        let ridge_candidate = RidgeCandidate::<3>::try_from_vertices([v0, v1]).unwrap();

        let images =
            simplex_lifted_ridge_vertex_images(&tds, simplex_key, &ridge_candidate).unwrap();

        assert_eq!(
            images.len(),
            2,
            "duplicate quotient vertex slots with different offsets should produce two lifted ridge images"
        );
        assert!(
            images.iter().any(|image| {
                image.contains(&LiftedVertexId::base(v0))
                    && image.contains(&LiftedVertexId::base(v1))
            }),
            "one lifted ridge image should use the base occurrence"
        );
        assert!(
            images.iter().any(|image| {
                image.contains(&LiftedVertexId::base(v0))
                    && image.contains(&lifted_vertex_id(v1, [-1_i16, 0, 0]))
            }),
            "one lifted ridge image should be normalized against the translated occurrence"
        );
    }

    #[test]
    fn test_normalized_simplex_vertices_for_lifted_target_enumerates_duplicate_anchor_slots() {
        let (tds, simplex_key, v0, v1, _v2) = build_duplicate_lifted_anchor_fixture_3d();
        let target: LiftedVertexBuffer = iter::once(LiftedVertexId::base(v0)).collect();

        let images =
            normalized_simplex_vertices_for_lifted_target(&tds, simplex_key, &target).unwrap();

        assert_eq!(
            images.len(),
            2,
            "normalization should consider both lifted occurrences of the target anchor"
        );
        assert!(
            images
                .iter()
                .any(|image| image.contains(&LiftedVertexId::base(v1))),
            "one simplex image should be anchored to the base v0 occurrence"
        );
        assert!(
            images
                .iter()
                .any(|image| image.contains(&lifted_vertex_id(v1, [-1_i16, 0, 0]))),
            "one simplex image should be anchored to the translated v0 occurrence"
        );
    }

    #[test]
    fn test_ridge_star_simplices_returns_incident_simplices_for_vertex_ridge_in_2d() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 1.0]))
            .unwrap();

        let c012 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let c013 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3], None).unwrap(),
            )
            .unwrap();
        let c023 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
            )
            .unwrap();
        let _c123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let ridge_candidate = RidgeCandidate::<2>::try_from_vertices([v0]).unwrap();
        let star = ridge_star_simplices(&tds, &ridge_candidate).unwrap();
        let star_set: SimplexKeySet = star.iter().copied().collect();

        let expected: SimplexKeySet = [c012, c013, c023].into_iter().collect();
        assert_eq!(star_set, expected);
    }

    #[test]
    fn test_ridge_star_simplices_returns_full_edge_star_in_3d() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();
        let v4 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, -1.0, 0.0]))
            .unwrap();

        let c0123 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let c0134 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v3, v4], None).unwrap(),
            )
            .unwrap();
        let c0142 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v4, v2], None).unwrap(),
            )
            .unwrap();

        let ridge_candidate = RidgeCandidate::<3>::try_from_vertices([v0, v1]).unwrap();
        let ridge_query = ridge_candidate.query(&tds).unwrap();
        let star = ridge_star_simplices(&tds, &ridge_candidate).unwrap();
        let query_star = ridge_query.incident_simplices();
        let star_set: SimplexKeySet = star.iter().copied().collect();
        let query_star_set: SimplexKeySet = query_star.iter().copied().collect();
        let ridge_view = ridge_candidate.view(&tds).unwrap();
        let view_star_set: SimplexKeySet =
            ridge_view.incident_simplices().iter().copied().collect();
        let query_vertex_uuids: SmallBuffer<_, 8> = ridge_query
            .vertices()
            .iter()
            .map(|vertex| vertex.uuid())
            .collect();
        let ridge_vertex_uuids: SmallBuffer<_, 8> = ridge_view
            .vertices()
            .iter()
            .map(|vertex| vertex.uuid())
            .collect();
        let ridge_links = ridge_view.links().unwrap();
        let ridge_link = ridge_links
            .first()
            .expect("non-periodic ridge should have one lifted link");
        let link_edge_set: FastHashSet<_> = ridge_link
            .edges()
            .iter()
            .map(LiftedLinkEdge::vertex_keys)
            .collect();
        let edge_pair = |a: VertexKey, b: VertexKey| {
            if a.data().as_ffi() <= b.data().as_ffi() {
                (a, b)
            } else {
                (b, a)
            }
        };

        let expected: SimplexKeySet = [c0123, c0134, c0142].into_iter().collect();
        assert_eq!(ridge_query.ridge_candidate(), &ridge_candidate);
        assert_eq!(ridge_query.vertex_keys(), ridge_candidate.as_slice());
        assert_eq!(star_set, expected);
        assert_eq!(query_star_set, expected);
        assert_eq!(view_star_set, expected);
        assert_eq!(ridge_view.ridge_candidate(), &ridge_candidate);
        assert_eq!(ridge_view.vertex_keys(), ridge_candidate.as_slice());
        assert_eq!(query_vertex_uuids, ridge_vertex_uuids);
        assert_eq!(ridge_vertex_uuids.len(), 2);
        assert_eq!(ridge_links.len(), 1);
        assert_eq!(ridge_link.quotient_ridge_candidate(), &ridge_candidate);
        assert_eq!(ridge_link.lifted_ridge_vertices().len(), 2);
        assert_eq!(
            link_edge_set,
            [edge_pair(v2, v3), edge_pair(v3, v4), edge_pair(v4, v2)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn test_ridge_star_simplices_errors_on_missing_vertex_key() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let missing = VertexKey::from(KeyData::from_ffi(u64::MAX));

        let ridge_candidate = RidgeCandidate::<2>::try_from_vertices([missing]).unwrap();
        match ridge_candidate.view(&tds) {
            Err(ManifoldError::Tds(TdsError::VertexNotFound {
                vertex_key,
                context,
            })) => {
                assert_eq!(vertex_key, missing);
                assert!(context.contains("ridge query"));
            }
            other => panic!("Expected ridge view VertexNotFound error, got {other:?}"),
        }
        match ridge_star_simplices(&tds, &ridge_candidate) {
            Err(ManifoldError::Tds(TdsError::VertexNotFound { vertex_key, .. })) => {
                assert_eq!(vertex_key, missing);
            }
            other => panic!("Expected VertexNotFound error, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_link_edges_from_star_rejects_self_loop_edge() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
            simplex.push_vertex_key(v1);
        }

        match ridge_link_edges_from_star(&tds, &simplex(&[v0]), &[simplex_key]) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { message })) => {
                assert!(
                    message.contains("self-loop"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected self-loop edge error, got {other:?}"),
        }
    }

    #[test]
    fn test_build_ridge_star_map_empty_returns_empty() {
        let tds: Tds<(), (), 3> = Tds::empty();

        let map = build_ridge_star_map(&tds).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_noop_for_d_lt_2() {
        let mut tds: Tds<(), (), 1> = Tds::empty();
        let v0 = tds.insert_vertex_with_mapping(test_vertex([0.0])).unwrap();
        let v1 = tds.insert_vertex_with_mapping(test_vertex([1.0])).unwrap();
        tds.insert_simplex_with_mapping(Simplex::try_new_with_data(vec![v0, v1], None).unwrap())
            .unwrap();

        let map = build_ridge_star_map(&tds).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_errors_on_corrupted_simplex_vertex_count() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        match build_ridge_star_map(&tds) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })) => {}
            other => {
                panic!("Expected DimensionMismatch(3, 2) for corrupted simplex, got {other:?}")
            }
        }
    }

    #[test]
    fn test_ridge_view_links_ignore_disjoint_corrupted_simplex() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let mut insert_vertex =
            |coords| tds.insert_vertex_with_mapping(test_vertex(coords)).unwrap();

        let v0 = insert_vertex([0.0, 0.0, 0.0]);
        let v1 = insert_vertex([1.0, 0.0, 0.0]);
        let v2 = insert_vertex([0.0, 1.0, 0.0]);
        let v3 = insert_vertex([0.0, 0.0, 1.0]);
        let u0 = insert_vertex([10.0, 0.0, 0.0]);
        let u1 = insert_vertex([11.0, 0.0, 0.0]);
        let u2 = insert_vertex([10.0, 1.0, 0.0]);
        let u3 = insert_vertex([10.0, 0.0, 1.0]);

        let target_simplex = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        let corrupted_simplex = tds
            .insert_simplex_bypassing_topology_checks_for_test(
                Simplex::try_new_with_data(vec![u0, u1, u2, u3], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds
                .simplex_mut(corrupted_simplex)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(u0);
            simplex.push_vertex_key(u1);
            simplex.push_vertex_key(u2);
        }

        match build_ridge_star_map(&tds) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 4,
                actual: 3,
                ..
            })) => {}
            other => panic!("Expected whole-TDS ridge map to fail, got {other:?}"),
        }

        let ridge_candidate = RidgeCandidate::<3>::try_from_vertices([v0, v1]).unwrap();
        let ridge_links = ridge_candidate.view(&tds).unwrap().links().unwrap();

        assert_eq!(ridge_links.len(), 1);
        assert_eq!(ridge_links[0].incident_simplices(), &[target_simplex]);
        assert_eq!(ridge_links[0].edges().len(), 1);
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_noop_for_d_lt_2() {
        let tds: Tds<(), (), 1> = Tds::empty();
        let simplex_key = SimplexKey::from(KeyData::from_ffi(0));

        let map = build_ridge_star_map_for_simplices(&tds, [simplex_key]).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_empty_returns_empty() {
        let mut tds: Tds<(), (), 3> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();

        let _ = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();

        let map = build_ridge_star_map_for_simplices(&tds, iter::empty::<SimplexKey>()).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_3d_single_simplex_includes_only_its_ridges_and_full_stars()
     {
        let (tds, [v0, v1, v2, v3, v4], [c1, c2]) = build_two_tetrahedra_sharing_facet_tds_3d();

        let missing = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let map = build_ridge_star_map_for_simplices(&tds, [c1, missing]).unwrap();

        assert_eq!(map.len(), 6);

        let star_set_for_edge = |a: VertexKey, b: VertexKey| -> SimplexKeySet {
            let key = facet_key_from_vertices(&[a, b]);
            let star = map
                .get(&key)
                .expect("expected ridge key in local ridge-star map");

            assert_eq!(periodic_simplex_key(&star.ridge_vertices), key);
            assert_eq!(star.ridge_vertices.len(), 2);

            star.star_simplices.iter().copied().collect()
        };

        let shared_star: SimplexKeySet = [c1, c2].into_iter().collect();
        let c1_only: SimplexKeySet = iter::once(c1).collect();

        assert_eq!(star_set_for_edge(v0, v1), shared_star);
        assert_eq!(star_set_for_edge(v0, v2), shared_star);
        assert_eq!(star_set_for_edge(v1, v2), shared_star);

        assert_eq!(star_set_for_edge(v0, v3), c1_only);
        assert_eq!(star_set_for_edge(v1, v3), c1_only);
        assert_eq!(star_set_for_edge(v2, v3), c1_only);

        assert!(!map.contains_key(&facet_key_from_vertices(&[v0, v4])));
        assert!(!map.contains_key(&facet_key_from_vertices(&[v1, v4])));
        assert!(!map.contains_key(&facet_key_from_vertices(&[v2, v4])));
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_3d_two_simplices_includes_union_of_ridges() {
        let (tds, [v0, v1, v2, v3, v4], [c1, c2]) = build_two_tetrahedra_sharing_facet_tds_3d();

        let map = build_ridge_star_map_for_simplices(&tds, [c1, c2]).unwrap();

        assert_eq!(map.len(), 9);

        let star_size_for_edge = |a: VertexKey, b: VertexKey| -> usize {
            let key = facet_key_from_vertices(&[a, b]);
            map.get(&key)
                .expect("expected ridge key in local ridge-star map")
                .star_simplices
                .len()
        };

        assert_eq!(star_size_for_edge(v0, v1), 2);
        assert_eq!(star_size_for_edge(v0, v2), 2);
        assert_eq!(star_size_for_edge(v1, v2), 2);

        assert_eq!(star_size_for_edge(v0, v3), 1);
        assert_eq!(star_size_for_edge(v1, v3), 1);
        assert_eq!(star_size_for_edge(v2, v3), 1);

        assert_eq!(star_size_for_edge(v0, v4), 1);
        assert_eq!(star_size_for_edge(v1, v4), 1);
        assert_eq!(star_size_for_edge(v2, v4), 1);
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_2d_includes_full_star_for_shared_vertex() {
        let (tds, v0, incident, _nonincident) = build_wedge_two_spheres_share_vertex_tds_2d();

        let map = build_ridge_star_map_for_simplices(&tds, [incident]).unwrap();
        assert_eq!(map.len(), 3);

        let ridge_key = facet_key_from_vertices(&[v0]);
        let star = map
            .get(&ridge_key)
            .expect("expected ridge key for shared vertex");
        assert_eq!(star.star_simplices.len(), 6);
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_errors_on_corrupted_simplex_vertex_count() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0]))
            .unwrap();

        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();

        {
            let simplex = tds
                .simplex_mut(simplex_key)
                .expect("simplex key should be valid in test");
            simplex.clear_vertex_keys();
            simplex.push_vertex_key(v0);
            simplex.push_vertex_key(v1);
        }

        match build_ridge_star_map_for_simplices(&tds, [simplex_key]) {
            Err(ManifoldError::Tds(TdsError::DimensionMismatch {
                expected: 3,
                actual: 2,
                ..
            })) => {}
            other => {
                panic!("Expected DimensionMismatch(3, 2) for corrupted simplex, got {other:?}")
            }
        }
    }

    #[test]
    fn test_simplex_star_simplices_rejects_missing_vertex() {
        let tds: Tds<(), (), 2> = Tds::empty();
        let stale_key = VertexKey::from(KeyData::from_ffi(0xDEAD));
        match simplex_star_simplices(&tds, &[stale_key]) {
            Err(ManifoldError::Tds(TdsError::VertexNotFound {
                vertex_key,
                ref context,
            })) => {
                assert_eq!(vertex_key, stale_key);
                assert!(context.contains("simplex star"));
            }
            other => panic!("Expected VertexNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_candidate_rejects_too_many_vertices_in_3d() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));
        let v1 = VertexKey::from(KeyData::from_ffi(2));
        let v2 = VertexKey::from(KeyData::from_ffi(3));
        match RidgeCandidate::<3>::try_from_vertices([v0, v1, v2]) {
            Err(RidgeCandidateError::WrongArity {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            other => panic!("Expected WrongArity, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_candidate_rejects_duplicate_vertices() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));

        match RidgeCandidate::<3>::try_from_vertices([v0, v0]) {
            Err(RidgeCandidateError::DuplicateVertex { vertex_key }) => {
                assert_eq!(vertex_key, v0);
            }
            other => panic!("Expected DuplicateVertex, got {other:?}"),
        }
    }

    #[test]
    fn test_ridge_candidate_canonicalizes_permuted_vertices() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));
        let v1 = VertexKey::from(KeyData::from_ffi(2));

        let forward = RidgeCandidate::<3>::try_from_vertices([v0, v1]).unwrap();
        let reversed = RidgeCandidate::<3>::try_from_vertices([v1, v0]).unwrap();

        assert_eq!(forward, reversed);
        assert_eq!(reversed.as_slice(), &[v0, v1]);
    }

    #[test]
    fn test_ridge_query_view_and_link_trait_behavior() {
        let (tds, [v0, v1, _v2, _v3, _v4], _) = build_two_tetrahedra_sharing_facet_tds_3d();

        let ridge_candidate = RidgeCandidate::<3>::try_from_vertices([v1, v0]).unwrap();
        assert_eq!(ridge_candidate.as_slice(), &[v0, v1]);

        let query = ridge_candidate.query(&tds).unwrap();
        let query_clone = query.clone();
        assert_eq!(query, query_clone);
        assert!(format!("{query:?}").contains("RidgeQuery"));

        let view = ridge_candidate.view(&tds).unwrap();
        let view_clone = view.clone();
        assert_eq!(view, view_clone);
        assert!(format!("{view:?}").contains("RidgeView"));

        let links = view.links().unwrap();
        assert!(!links.is_empty());

        let link = links[0].clone();
        assert_eq!(link, link.clone());
        assert_eq!(link.quotient_ridge_candidate(), view.ridge_candidate());
        assert!(!link.incident_simplices().is_empty());
        assert!(!link.edges().is_empty());
        assert!(format!("{link:?}").contains("RidgeLinkView"));
    }

    #[test]
    fn test_build_ridge_star_map_for_simplices_identifies_translated_periodic_images() {
        let mut tds: Tds<(), (), 2> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.5, 1.0]))
            .unwrap();

        let mut simplex1 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex1
            .set_periodic_vertex_offsets(vec![[0, 0], [0, 0], [0, 0]])
            .unwrap();
        let c1 = tds.insert_simplex_with_mapping(simplex1).unwrap();

        let mut simplex2 = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex2
            .set_periodic_vertex_offsets(vec![[1, 0], [0, 0], [0, 0]])
            .unwrap();
        let c2 = tds.insert_simplex_with_mapping(simplex2).unwrap();

        let map = build_ridge_star_map_for_simplices(&tds, [c1, c2]).unwrap();

        assert_eq!(map.len(), 3, "expected 3 quotient-aware ridges");

        let shared_count = map.values().filter(|s| s.star_simplices.len() == 2).count();
        assert_eq!(shared_count, 3, "three ridges should be shared");

        let ridge_candidate = RidgeCandidate::<2>::try_from_vertices([v0]).unwrap();
        let ridge_view = ridge_candidate.view(&tds).unwrap();
        let ridge_links = ridge_view.links().unwrap();
        assert_eq!(
            ridge_links.len(),
            1,
            "single-vertex translated ridges normalize to one lifted link"
        );

        let link_edges = ridge_links[0].edges();
        assert_eq!(link_edges.len(), 2);

        let quotient_edges: FastHashSet<_> =
            link_edges.iter().map(LiftedLinkEdge::vertex_keys).collect();
        assert_eq!(
            quotient_edges.len(),
            1,
            "bare vertex keys collapse the two periodic link edges"
        );

        let lifted_offsets: FastHashSet<_> = link_edges
            .iter()
            .map(|edge| {
                let (a, b) = edge.endpoints();
                (a.offset().to_vec(), b.offset().to_vec())
            })
            .collect();
        assert_eq!(
            lifted_offsets,
            [
                (Vec::<i16>::new(), Vec::<i16>::new()),
                (vec![-1_i16, 0], vec![-1_i16, 0]),
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn test_periodic_aware_ridge_star_empty_star_returns_error() {
        let mut tds: Tds<(), (), 3> = Tds::empty();

        let v0 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 0.0]))
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(test_vertex([1.0, 0.0, 0.0]))
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 1.0, 0.0]))
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(test_vertex([0.0, 0.0, 1.0]))
            .unwrap();

        let c1 = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2, v3], None).unwrap(),
            )
            .unwrap();
        tds.simplex_mut(c1)
            .unwrap()
            .set_periodic_vertex_offsets(vec![[0, 0, 0]; 4])
            .unwrap();

        let bare: VertexKeyBuffer = [v0, v1].into_iter().collect();
        let lifted: LiftedVertexBuffer = [
            LiftedVertexId::base(v0),
            lifted_vertex_id(v1, [99_i16, 99_i16, 99_i16]),
        ]
        .into_iter()
        .collect();

        match periodic_aware_ridge_star(&tds, 0x42, &lifted, &bare) {
            Err(ManifoldError::Tds(TdsError::InconsistentDataStructure { ref message })) => {
                assert!(
                    message.contains("empty star"),
                    "error should mention empty star: {message}"
                );
            }
            other => panic!("Expected InconsistentDataStructure (empty star), got {other:?}"),
        }
    }
}
