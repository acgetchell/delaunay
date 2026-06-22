//! Toroidal topological space and periodic covering-space identities.
//!
//! This module defines [`ToroidalSpace`] and the lifted runtime identities used
//! when topology validators inspect periodic triangulations. [`LiftedVertexId`]
//! and [`LiftedLinkEdge`] are not TDS storage keys; they identify images in a
//! local covering-space frame so ridge-link and vertex-link checks can preserve
//! toroidal adjacency instead of collapsing immediately to quotient
//! [`VertexKey`](crate::core::tds::VertexKey) values.

#![forbid(unsafe_code)]

use crate::core::{
    collections::{FastHasher, SmallBuffer, VertexKeyBuffer},
    facet::facet_key_from_vertices,
    tds::VertexKey,
};
use crate::topology::traits::topological_space::{
    TopologicalSpace, TopologyKind, ToroidalDomain, ToroidalDomainError,
};
use slotmap::Key;
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

// =============================================================================
// Periodic covering-space identities
// =============================================================================

/// Vertex identity in a periodic covering space.
///
/// This deliberately is not a `VertexKey`: lifted periodic images are graph
/// identities used by topology validators, not entries in the TDS vertex store.
/// The value is runtime-local because it contains a storage-local [`VertexKey`].
///
/// Callers usually obtain these values from
/// [`crate::topology::ridge::RidgeLinkView::lifted_ridge_vertices`] or from
/// [`LiftedLinkEdge::endpoints`]. Use [`Self::vertex_key`] only when explicitly
/// choosing quotient-space semantics.
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
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.tds().vertex_keys().take(1))?;
/// let view = ridge.view(dt.tds())?;
/// let links = view.links()?;
/// let Some(edge) = links.first().and_then(|link| link.edges().first()) else {
///     return Ok(());
/// };
/// let (endpoint, _) = edge.endpoints();
///
/// assert_eq!(endpoint.vertex_key(), edge.vertex_keys().0);
/// assert!(endpoint.is_base());
/// assert!(endpoint.offset().is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiftedVertexId {
    pub(crate) vertex_key: VertexKey,
    offset: SmallBuffer<i16, 8>,
}

pub(crate) type LiftedVertexBuffer = SmallBuffer<LiftedVertexId, 8>;
pub(crate) type LinkSimplexBuffer = SmallBuffer<LiftedVertexBuffer, 8>;

impl LiftedVertexId {
    /// Creates the base periodic image for a quotient-space vertex key.
    pub(crate) fn base(vertex_key: VertexKey) -> Self {
        Self {
            vertex_key,
            offset: SmallBuffer::new(),
        }
    }

    /// Returns the quotient-space vertex key represented by this lifted image.
    #[inline]
    #[must_use]
    pub const fn vertex_key(&self) -> VertexKey {
        self.vertex_key
    }

    /// Returns the periodic lattice offset for this lifted image.
    ///
    /// An empty slice means the base image. Offsets are interpreted relative to
    /// the local anchor used by the topology query that produced this value.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> &[i16] {
        &self.offset
    }

    /// Returns whether this is the base image of its quotient vertex.
    #[inline]
    #[must_use]
    pub fn is_base(&self) -> bool {
        self.offset.is_empty()
    }
}

/// Edge in a lifted link whose endpoints preserve periodic image identity.
///
/// `LiftedLinkEdge` is a runtime toroidal-topology value, not a durable
/// identifier. It may contain two endpoints with the same quotient
/// [`VertexKey`] but different periodic offsets; callers that collapse it to
/// bare keys are explicitly choosing quotient-space semantics.
///
/// These edges are produced by [`crate::topology::ridge::RidgeLinkView::edges`].
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
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
///
/// let ridge = RidgeCandidate::<2>::try_from_vertices(dt.tds().vertex_keys().take(1))?;
/// let view = ridge.view(dt.tds())?;
/// let links = view.links()?;
/// let Some(edge) = links.first().and_then(|link| link.edges().first()) else {
///     return Ok(());
/// };
///
/// let (first, second) = edge.endpoints();
/// assert_eq!(edge.vertex_keys(), (first.vertex_key(), second.vertex_key()));
/// assert!(!edge.is_quotient_self_loop());
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LiftedLinkEdge {
    endpoints: (LiftedVertexId, LiftedVertexId),
}

impl LiftedLinkEdge {
    pub(crate) fn from_unordered_endpoints(a: &LiftedVertexId, b: &LiftedVertexId) -> Self {
        Self {
            endpoints: ordered_lifted_edge(a, b),
        }
    }

    /// Returns the lifted endpoints in canonical order.
    #[inline]
    #[must_use]
    pub const fn endpoints(&self) -> (&LiftedVertexId, &LiftedVertexId) {
        (&self.endpoints.0, &self.endpoints.1)
    }

    /// Returns the quotient-space endpoint keys.
    ///
    /// This intentionally discards periodic image identity. Use
    /// [`Self::endpoints`] when lifted topology matters.
    #[inline]
    #[must_use]
    pub const fn vertex_keys(&self) -> (VertexKey, VertexKey) {
        (self.endpoints.0.vertex_key, self.endpoints.1.vertex_key)
    }

    /// Returns whether the lifted edge collapses to one quotient-space vertex.
    #[inline]
    #[must_use]
    pub fn is_quotient_self_loop(&self) -> bool {
        self.endpoints.0.vertex_key == self.endpoints.1.vertex_key
    }
}

impl Ord for LiftedVertexId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.vertex_key
            .data()
            .as_ffi()
            .cmp(&other.vertex_key.data().as_ffi())
            .then_with(|| self.offset.as_slice().cmp(other.offset.as_slice()))
    }
}

impl PartialOrd for LiftedVertexId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for LiftedVertexId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_key.data().as_ffi().hash(state);
        self.offset.as_slice().hash(state);
    }
}

/// Creates a lifted vertex identity from a real TDS vertex key and periodic
/// lattice offset.
///
/// Zero offsets are normalized to the base image so base-image identity remains
/// compact and comparisons do not depend on explicit all-zero offset storage.
pub(crate) fn lifted_vertex_id(
    vk: VertexKey,
    offset: impl IntoIterator<Item = i16>,
) -> LiftedVertexId {
    let mut offset_buffer: SmallBuffer<i16, 8> = SmallBuffer::new();
    let mut has_nonzero_component = false;
    for component in offset {
        has_nonzero_component |= component != 0;
        offset_buffer.push(component);
    }
    if !has_nonzero_component {
        return LiftedVertexId::base(vk);
    }
    LiftedVertexId {
        vertex_key: vk,
        offset: offset_buffer,
    }
}

/// Computes a periodic-aware simplex key from lifted vertex IDs.
///
/// The key is translation-invariant in the covering space: translating every
/// lifted vertex by the same lattice offset produces the same key. Base images
/// reuse [`facet_key_from_vertices`] so non-periodic topology keeps the same
/// hash path as quotient-space facets.
pub(crate) fn periodic_simplex_key(lifted_vertices: &[LiftedVertexId]) -> u64 {
    if lifted_vertices.iter().all(LiftedVertexId::is_base) {
        let bare_vertices: VertexKeyBuffer =
            lifted_vertices.iter().map(|id| id.vertex_key).collect();
        return facet_key_from_vertices(&bare_vertices);
    }

    let keys = normalize_lifted_vertices(lifted_vertices);
    let mut hasher = FastHasher::default();
    for key in &keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Computes an exact lifted simplex key without quotient translation normalization.
///
/// Vertex links already express every lifted vertex relative to the linked
/// anchor so applying an additional global translation quotient can collapse
/// distinct link simplices.
pub(crate) fn anchored_lifted_simplex_key(lifted_vertices: &[LiftedVertexId]) -> u64 {
    if lifted_vertices.iter().all(LiftedVertexId::is_base) {
        let bare_vertices: VertexKeyBuffer =
            lifted_vertices.iter().map(|id| id.vertex_key).collect();
        return facet_key_from_vertices(&bare_vertices);
    }

    let mut keys: LiftedVertexBuffer = lifted_vertices.iter().cloned().collect();
    keys.sort_unstable();
    let mut hasher = FastHasher::default();
    for key in &keys {
        key.hash(&mut hasher);
    }
    hasher.finish()
}

/// Normalizes lifted vertices by subtracting the offset of the first sorted
/// lifted vertex, making periodic simplex identities translation invariant.
pub(crate) fn normalize_lifted_vertices(lifted_vertices: &[LiftedVertexId]) -> LiftedVertexBuffer {
    let mut keys: LiftedVertexBuffer = lifted_vertices.iter().cloned().collect();
    keys.sort_unstable();
    let anchor_offset: SmallBuffer<i16, 8> = keys
        .first()
        .map_or_else(SmallBuffer::new, |key| key.offset.clone());
    let axes = keys
        .iter()
        .map(|key| key.offset.len())
        .max()
        .unwrap_or(0)
        .max(anchor_offset.len());

    let mut normalized = LiftedVertexBuffer::with_capacity(keys.len());
    for key in keys {
        let mut offset: SmallBuffer<i16, 8> = SmallBuffer::with_capacity(axes);
        for axis in 0..axes {
            let component = key.offset.get(axis).copied().unwrap_or(0)
                - anchor_offset.get(axis).copied().unwrap_or(0);
            offset.push(component);
        }
        normalized.push(lifted_vertex_id(key.vertex_key, offset));
    }
    normalized
}

/// Returns a canonical ordering for a lifted link edge.
///
/// Ordering includes both the quotient [`VertexKey`] and the periodic offset,
/// so quotient self-loops with distinct lifted endpoints remain distinct.
pub(crate) fn ordered_lifted_edge(
    a: &LiftedVertexId,
    b: &LiftedVertexId,
) -> (LiftedVertexId, LiftedVertexId) {
    if b < a {
        (b.clone(), a.clone())
    } else {
        (a.clone(), b.clone())
    }
}

/// Represents toroidal topological space with periodic boundaries.
///
/// Toroidal spaces have periodic boundary conditions defined by a
/// fundamental domain. For example, a 2-torus (T²) has Euler
/// characteristic χ = 0.
///
/// The dimension `D` is a const generic parameter that must match the
/// dimension of the associated triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::{ToroidalDomain, ToroidalSpace};
///
/// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
/// let domain = ToroidalDomain::<2>::try_new([1.0, 2.0])?;
/// let space = ToroidalSpace::new(domain);
/// assert_eq!(space.domain().periods(), &[1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ToroidalSpace<const D: usize> {
    /// The fundamental domain defining the period of each dimension.
    domain: ToroidalDomain<D>,
}

impl<const D: usize> ToroidalSpace<D> {
    /// Creates a new toroidal space from a validated fundamental domain.
    ///
    /// # Arguments
    ///
    /// * `domain` - Validated periods for periodic boundary conditions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{ToroidalDomain, ToroidalSpace};
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
    /// let domain = ToroidalDomain::<2>::try_new([2.0, 3.0])?;
    /// let space = ToroidalSpace::new(domain);
    /// assert_eq!(space.domain().periods(), &[2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn new(domain: ToroidalDomain<D>) -> Self {
        Self { domain }
    }

    /// Creates a new toroidal space from raw fundamental-domain periods.
    ///
    /// # Errors
    ///
    /// Returns [`ToroidalDomainError::InvalidPeriod`] when any period is
    /// non-finite, zero, or negative.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::{ToroidalDomainError, ToroidalSpace};
    ///
    /// # fn main() -> Result<(), ToroidalDomainError> {
    /// let space = ToroidalSpace::<2>::try_new([1.0, 2.0])?;
    /// assert_eq!(space.domain().periods(), &[1.0, 2.0]);
    ///
    /// std::assert_matches!(
    ///     ToroidalSpace::<2>::try_new([f64::NAN, 2.0]),
    ///     Err(ToroidalDomainError::InvalidPeriod { axis: 0, period })
    ///         if period.is_nan()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(domain: [f64; D]) -> Result<Self, ToroidalDomainError> {
        Ok(Self::new(ToroidalDomain::try_new(domain)?))
    }

    /// Creates a unit toroidal space where every dimension has period 1.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::ToroidalSpace;
    ///
    /// let space = ToroidalSpace::<3>::unit();
    /// assert_eq!(space.domain().periods(), &[1.0, 1.0, 1.0]);
    /// ```
    #[must_use]
    pub const fn unit() -> Self {
        Self {
            domain: ToroidalDomain::unit(),
        }
    }

    /// Returns the validated fundamental domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::ToroidalSpace;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
    /// let space = ToroidalSpace::<2>::try_new([2.0, 3.0])?;
    /// assert_eq!(space.domain().period(1), Some(3.0));
    /// # Ok(())
    /// # }
    /// ```
    pub const fn domain(&self) -> &ToroidalDomain<D> {
        &self.domain
    }

    /// Wraps a single coordinate value into the fundamental domain `[0, L_axis)`
    /// using `rem_euclid` arithmetic.
    ///
    /// Applies `rem_euclid(domain[axis])`. Returns `None` if `axis` is out of
    /// range or `value` is not finite.
    ///
    /// # Arguments
    ///
    /// * `axis` - The dimension index (must be `< D`; returns `None` if out of range).
    /// * `value` - The coordinate value to wrap.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::ToroidalSpace;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::topology::spaces::ToroidalDomainError> {
    /// let space = ToroidalSpace::<2>::try_new([1.0, 2.0])?;
    ///
    /// // Positive out-of-range
    /// assert_eq!(space.wrap_coord(0, 1.7), Some(0.7));
    ///
    /// // Negative wraps to positive
    /// assert_eq!(space.wrap_coord(1, -0.5), Some(1.5));
    ///
    /// // Out-of-range axis returns None
    /// assert_eq!(space.wrap_coord(5, 0.3), None);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn wrap_coord(&self, axis: usize, value: f64) -> Option<f64> {
        let period = self.domain.period(axis)?;
        if !value.is_finite() {
            return None;
        }
        Some(value.rem_euclid(period))
    }
}

impl<const D: usize> TopologicalSpace for ToroidalSpace<D> {
    const DIM: usize = D;

    fn kind(&self) -> TopologyKind {
        TopologyKind::Toroidal
    }

    fn allows_boundary(&self) -> bool {
        false
    }

    fn canonicalize_point(&self, coords: &mut [f64]) {
        for (coord, &period) in coords.iter_mut().zip(self.domain.periods().iter()) {
            *coord = coord.rem_euclid(period);
        }
    }

    fn fundamental_domain(&self) -> Option<&[f64]> {
        Some(&self.domain.periods()[..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use slotmap::KeyData;

    #[test]
    fn test_anchored_lifted_simplex_key_preserves_vertex_link_offsets() {
        let v0 = VertexKey::from(KeyData::from_ffi(1));
        let v1 = VertexKey::from(KeyData::from_ffi(2));
        let v2 = VertexKey::from(KeyData::from_ffi(3));

        let first_link_triangle: LiftedVertexBuffer = [
            lifted_vertex_id(v0, [1_i16, 0, 0]),
            lifted_vertex_id(v1, [1_i16, 0, 0]),
            lifted_vertex_id(v2, [1_i16, 0, 0]),
        ]
        .into_iter()
        .collect();
        let shifted_link_triangle: LiftedVertexBuffer = [
            lifted_vertex_id(v0, [2_i16, 0, 0]),
            lifted_vertex_id(v1, [2_i16, 0, 0]),
            lifted_vertex_id(v2, [2_i16, 0, 0]),
        ]
        .into_iter()
        .collect();

        assert_eq!(
            periodic_simplex_key(&first_link_triangle),
            periodic_simplex_key(&shifted_link_triangle),
            "quotient simplex keys intentionally identify global translations"
        );
        assert_ne!(
            anchored_lifted_simplex_key(&first_link_triangle),
            anchored_lifted_simplex_key(&shifted_link_triangle),
            "vertex-link keys must preserve offsets relative to the linked vertex"
        );
    }

    #[test]
    fn test_lifted_vertex_id_normalizes_zero_offsets_to_base_image() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(1));

        let explicit_zero = lifted_vertex_id(vertex_key, [0_i16, 0, 0]);
        let shifted = lifted_vertex_id(vertex_key, [0_i16, 1, 0]);

        assert_eq!(explicit_zero.vertex_key(), vertex_key);
        assert!(explicit_zero.is_base());
        assert!(explicit_zero.offset().is_empty());
        assert_eq!(shifted.vertex_key(), vertex_key);
        assert!(!shifted.is_base());
        assert_eq!(shifted.offset(), &[0, 1, 0]);
    }

    #[test]
    fn test_lifted_link_edge_preserves_periodic_self_loop_identity() {
        let vertex_key = VertexKey::from(KeyData::from_ffi(1));
        let base = lifted_vertex_id(vertex_key, [0_i16, 0]);
        let shifted = lifted_vertex_id(vertex_key, [1_i16, 0]);

        let edge = LiftedLinkEdge::from_unordered_endpoints(&shifted, &base);
        let (first, second) = edge.endpoints();

        assert!(edge.is_quotient_self_loop());
        assert_eq!(edge.vertex_keys(), (vertex_key, vertex_key));
        assert_eq!(first, &base);
        assert_eq!(second, &shifted);
        assert_ne!(first.offset(), second.offset());
    }

    #[test]
    fn test_new() {
        let domain = ToroidalDomain::try_new([1.0, 2.0, 3.0]).unwrap();
        let space = ToroidalSpace::<3>::new(domain);
        assert_eq!(ToroidalSpace::<3>::DIM, 3);
        assert_relative_eq!(space.domain().periods()[0], 1.0);
        assert_relative_eq!(space.domain().periods()[1], 2.0);
        assert_relative_eq!(space.domain().periods()[2], 3.0);
    }

    #[test]
    fn test_try_new_rejects_invalid_domain() {
        let err = ToroidalSpace::<2>::try_new([0.0, 1.0]).unwrap_err();
        assert_eq!(
            err,
            ToroidalDomainError::InvalidPeriod {
                axis: 0,
                period: 0.0,
            }
        );
    }

    #[test]
    fn test_kind() {
        let space = ToroidalSpace::<3>::unit();
        assert_eq!(space.kind(), TopologyKind::Toroidal);
    }

    #[test]
    fn test_allows_boundary() {
        let space = ToroidalSpace::<3>::unit();
        assert!(
            !space.allows_boundary(),
            "Toroidal space is a closed manifold with periodic boundaries"
        );
    }

    #[test]
    fn test_canonicalize_point() {
        let space = ToroidalSpace::<3>::try_new([2.0, 3.0, 4.0]).unwrap();
        let mut coords = [2.5, -1.0, 5.5];
        space.canonicalize_point(&mut coords);
        // 2.5 rem_euclid 2.0 = 0.5
        assert_relative_eq!(coords[0], 0.5);
        // -1.0 rem_euclid 3.0 = 2.0
        assert_relative_eq!(coords[1], 2.0);
        // 5.5 rem_euclid 4.0 = 1.5
        assert_relative_eq!(coords[2], 1.5);
    }

    #[test]
    fn test_canonicalize_point_idempotent() {
        let space = ToroidalSpace::<2>::unit();
        let mut coords = [0.3, 0.7];
        space.canonicalize_point(&mut coords);
        // Already in [0, 1), should be unchanged
        assert_relative_eq!(coords[0], 0.3);
        assert_relative_eq!(coords[1], 0.7);
        // Applying again should be unchanged (idempotent)
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.3);
        assert_relative_eq!(coords[1], 0.7);
    }

    #[test]
    fn test_canonicalize_point_boundary() {
        let space = ToroidalSpace::<2>::unit();
        // Exactly on boundary: 1.0 rem_euclid 1.0 = 0.0
        let mut coords = [1.0, 2.0];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.0);
        assert_relative_eq!(coords[1], 0.0);
    }

    #[test]
    fn test_canonicalize_point_negative() {
        let space = ToroidalSpace::<3>::unit();
        // Negative coordinates should wrap into [0, 1)
        let mut coords = [-0.1, -1.0, -2.5];
        space.canonicalize_point(&mut coords);
        assert_relative_eq!(coords[0], 0.9);
        assert_relative_eq!(coords[1], 0.0);
        assert_relative_eq!(coords[2], 0.5);
    }

    #[test]
    fn test_fundamental_domain() {
        let domain = [2.0, 3.0, 4.0];
        let space = ToroidalSpace::<3>::try_new(domain).unwrap();
        assert_eq!(space.fundamental_domain(), Some(&domain[..]));
    }

    #[test]
    fn test_different_domains() {
        // 2D unit square torus
        let unit_torus = ToroidalSpace::<2>::unit();
        assert_eq!(unit_torus.fundamental_domain(), Some(&[1.0, 1.0][..]));

        // 2D rectangular torus
        let rect_torus = ToroidalSpace::<2>::try_new([2.0, 3.0]).unwrap();
        assert_eq!(rect_torus.fundamental_domain(), Some(&[2.0, 3.0][..]));

        // 3D cube torus
        let cube_torus = ToroidalSpace::<3>::unit();
        assert_eq!(cube_torus.fundamental_domain(), Some(&[1.0, 1.0, 1.0][..]));
    }

    #[test]
    fn test_dimension_consistency() {
        assert_eq!(ToroidalSpace::<2>::DIM, 2);
        assert_eq!(ToroidalSpace::<3>::DIM, 3);
        assert_eq!(ToroidalSpace::<4>::DIM, 4);
        assert_eq!(ToroidalSpace::<5>::DIM, 5);
    }

    #[test]
    fn test_unit() {
        let space = ToroidalSpace::<3>::unit();
        assert_relative_eq!(space.domain().periods()[0], 1.0);
        assert_relative_eq!(space.domain().periods()[1], 1.0);
        assert_relative_eq!(space.domain().periods()[2], 1.0);
        let space2d = ToroidalSpace::<2>::unit();
        assert_relative_eq!(space2d.domain().periods()[0], 1.0);
        assert_relative_eq!(space2d.domain().periods()[1], 1.0);
    }

    #[test]
    fn test_wrap_coord_positive_out_of_range() {
        let space = ToroidalSpace::<2>::try_new([1.0, 2.0]).unwrap();
        let wrapped = space.wrap_coord(0, 1.7);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.7);
    }

    #[test]
    fn test_wrap_coord_negative() {
        let space = ToroidalSpace::<2>::try_new([1.0, 2.0]).unwrap();
        // -0.5 rem_euclid 2.0 = 1.5
        let wrapped = space.wrap_coord(1, -0.5);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 1.5);
    }

    #[test]
    fn test_wrap_coord_in_range_unchanged() {
        let space = ToroidalSpace::<2>::unit();
        let wrapped = space.wrap_coord(0, 0.3);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.3);
    }

    #[test]
    fn test_wrap_coord_boundary() {
        let space = ToroidalSpace::<2>::unit();
        // Exactly at period boundary wraps to 0
        let wrapped = space.wrap_coord(0, 1.0);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.0);
    }

    #[test]
    fn test_wrap_coord_out_of_range_axis() {
        let space = ToroidalSpace::<2>::unit();
        assert!(space.wrap_coord(5, 0.3).is_none());
    }

    #[test]
    fn test_wrap_coord_f64() {
        let space = ToroidalSpace::<2>::unit();
        let wrapped = space.wrap_coord(0, 1.5);
        assert!(wrapped.is_some());
        assert_relative_eq!(wrapped.unwrap(), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_wrap_coord_non_finite() {
        let space = ToroidalSpace::<2>::unit();
        assert!(space.wrap_coord(0, f64::NAN).is_none());
        assert!(space.wrap_coord(0, f64::INFINITY).is_none());
    }
}
