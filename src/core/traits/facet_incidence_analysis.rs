//! TDS-level facet-incidence analysis trait.

use crate::core::{
    facet::{FacetToSimplicesIndex, FacetView, OneSidedFacetsIter},
    tds::TdsError,
};

/// Trait for TDS-level facet-incidence queries.
///
/// The triangulation data structure can classify facet multiplicity: a facet is
/// incident to one or two D-simplices. That is deliberately weaker than manifold
/// boundary semantics. A one-sided facet can be a Euclidean boundary facet, but
/// in a periodic quotient triangulation it can also be a closed
/// self-identification. Public owner-level callers can inspect raw incidence
/// through [`Triangulation::facet_incidence_index`] or
/// [`DelaunayTriangulation::facet_incidence_index`] and
/// [`FacetIncidenceView::is_one_sided`]. Use
/// [`Triangulation::boundary_facets`] or
/// [`DelaunayTriangulation::boundary_facets`] for topology-aware boundary
/// queries.
///
/// [`Triangulation::facet_incidence_index`]: crate::Triangulation::facet_incidence_index
/// [`DelaunayTriangulation::facet_incidence_index`]: crate::DelaunayTriangulation::facet_incidence_index
/// [`FacetIncidenceView::is_one_sided`]: crate::tds::FacetIncidenceView::is_one_sided
/// [`Triangulation::boundary_facets`]: crate::Triangulation::boundary_facets
/// [`DelaunayTriangulation::boundary_facets`]: crate::DelaunayTriangulation::boundary_facets
pub trait FacetIncidenceAnalysis<U, V, const D: usize> {
    /// Identifies all one-sided facet incidences in the TDS.
    ///
    /// Implementations may build and sort a derived handle list so iteration is
    /// deterministic. This remains incidence analysis only; callers that need
    /// semantic boundary facets should use the topology-aware boundary APIs.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet-incidence index cannot be built.
    /// Individual iterator items return [`FacetError`](crate::prelude::tds::FacetError)
    /// if a facet view cannot be constructed from the live TDS.
    fn one_sided_facets(&self) -> Result<OneSidedFacetsIter<'_, U, V, D>, TdsError>;

    /// Checks whether a facet has one-sided incidence.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet-index cannot be built or if the
    /// supplied facet view belongs to a different TDS.
    fn is_one_sided_facet(&self, facet: &FacetView<'_, U, V, D>) -> Result<bool, TdsError>;

    /// Checks whether a facet has one-sided incidence using a prebuilt index.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the supplied facet view or facet-incidence index
    /// belongs to a different TDS.
    fn is_one_sided_facet_with_index(
        &self,
        facet: &FacetView<'_, U, V, D>,
        facet_to_simplices: &FacetToSimplicesIndex<'_, U, V, D>,
    ) -> Result<bool, TdsError>;

    /// Returns the number of one-sided facet incidences in the TDS.
    ///
    /// # Errors
    ///
    /// Returns a [`TdsError`] if the facet-incidence index cannot be built.
    fn number_of_one_sided_facets(&self) -> Result<usize, TdsError>;
}
