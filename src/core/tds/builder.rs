//! Fluent construction of proof-bearing triangulation data structures.
//!
//! [`TdsBuilder`] owns the raw explicit-connectivity boundary. It assembles
//! vertices and maximal simplices, derives adjacency and incidence, normalizes
//! coherent combinatorial orientation, and publishes a [`Tds`] only after
//! cumulative Levels 1–2 validation succeeds.

#![forbid(unsafe_code)]

use super::{
    Tds, TdsConstructionError, TdsDraft, TdsDraftError, TdsError, TdsMutationError, VertexKey,
};
use crate::core::collections::{
    Entry, FastHashMap, SimplexVertexKeyBuffer, fast_hash_map_with_capacity,
};
use crate::core::facet::facet_key_from_vertices;
use crate::core::simplex::{Simplex, SimplexValidationError};
use crate::core::vertex::Vertex;
use std::marker::PhantomData;
use thiserror::Error;

/// Parse failures at the raw explicit-connectivity boundary.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum ExplicitSimplexParseError {
    /// No maximal simplices were supplied for a nonempty vertex set.
    #[error("no simplices provided for nonempty TDS construction")]
    EmptySimplices,
    /// A simplex specification has the wrong arity for dimension `D`.
    #[error(
        "simplex {simplex_index} has {actual} vertex indices, expected {expected} for a simplex"
    )]
    InvalidSimplexArity {
        simplex_index: usize,
        actual: usize,
        expected: usize,
    },
    /// A simplex specification references a missing input vertex.
    #[error(
        "simplex {simplex_index} references vertex index {vertex_index}, but the vertex count is {bound}"
    )]
    IndexOutOfBounds {
        simplex_index: usize,
        vertex_index: usize,
        bound: usize,
    },
    /// A simplex specification repeats one input vertex.
    #[error("simplex {simplex_index} contains duplicate vertex index {vertex_index}")]
    DuplicateVertexInSimplex {
        simplex_index: usize,
        vertex_index: usize,
    },
}

impl From<ExplicitSimplexParseError> for TdsBuilderError {
    fn from(source: ExplicitSimplexParseError) -> Self {
        match source {
            ExplicitSimplexParseError::EmptySimplices => Self::EmptySimplices,
            ExplicitSimplexParseError::InvalidSimplexArity {
                simplex_index,
                actual,
                expected,
            } => Self::InvalidSimplexArity {
                simplex_index,
                actual,
                expected,
            },
            ExplicitSimplexParseError::IndexOutOfBounds {
                simplex_index,
                vertex_index,
                bound,
            } => Self::IndexOutOfBounds {
                simplex_index,
                vertex_index,
                bound,
            },
            ExplicitSimplexParseError::DuplicateVertexInSimplex {
                simplex_index,
                vertex_index,
            } => Self::DuplicateVertexInSimplex {
                simplex_index,
                vertex_index,
            },
        }
    }
}

/// Parsed explicit TDS input whose simplex indices are locally well formed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParsedTdsInput<'a, U, const D: usize> {
    vertices: &'a [Vertex<U, D>],
    simplices: &'a [Vec<usize>],
}

impl<'a, U, const D: usize> ParsedTdsInput<'a, U, D> {
    /// Parses raw vertex-index connectivity exactly once at the input boundary.
    pub(crate) fn try_new(
        vertices: &'a [Vertex<U, D>],
        simplices: &'a [Vec<usize>],
    ) -> Result<Self, ExplicitSimplexParseError> {
        if simplices.is_empty() && !vertices.is_empty() {
            return Err(ExplicitSimplexParseError::EmptySimplices);
        }

        for (simplex_index, simplex) in simplices.iter().enumerate() {
            if simplex.len() != D + 1 {
                return Err(ExplicitSimplexParseError::InvalidSimplexArity {
                    simplex_index,
                    actual: simplex.len(),
                    expected: D + 1,
                });
            }

            for (offset, &vertex_index) in simplex.iter().enumerate() {
                if vertex_index >= vertices.len() {
                    return Err(ExplicitSimplexParseError::IndexOutOfBounds {
                        simplex_index,
                        vertex_index,
                        bound: vertices.len(),
                    });
                }
                if simplex[..offset].contains(&vertex_index) {
                    return Err(ExplicitSimplexParseError::DuplicateVertexInSimplex {
                        simplex_index,
                        vertex_index,
                    });
                }
            }
        }

        Ok(Self {
            vertices,
            simplices,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum TdsBuilderInput<'a, U, const D: usize> {
    Raw {
        vertices: &'a [Vertex<U, D>],
        simplices: &'a [Vec<usize>],
    },
    Parsed(ParsedTdsInput<'a, U, D>),
}

impl<'a, U, const D: usize> TdsBuilderInput<'a, U, D> {
    const fn vertices(&self) -> &'a [Vertex<U, D>] {
        match self {
            Self::Raw { vertices, .. } | Self::Parsed(ParsedTdsInput { vertices, .. }) => vertices,
        }
    }

    const fn simplices(&self) -> &'a [Vec<usize>] {
        match self {
            Self::Raw { simplices, .. } | Self::Parsed(ParsedTdsInput { simplices, .. }) => {
                simplices
            }
        }
    }

    fn parse(self) -> Result<ParsedTdsInput<'a, U, D>, ExplicitSimplexParseError> {
        match self {
            Self::Raw {
                vertices,
                simplices,
            } => ParsedTdsInput::try_new(vertices, simplices),
            Self::Parsed(parsed) => Ok(parsed),
        }
    }
}

/// Typed failures from explicit Levels 1–2 TDS construction.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TdsBuilderError {
    /// No maximal simplices were supplied for a nonempty vertex set.
    #[error("no simplices provided for nonempty TDS construction")]
    EmptySimplices,
    /// A simplex specification has the wrong arity for dimension `D`.
    #[error(
        "simplex {simplex_index} has {actual} vertex indices, expected {expected} for a simplex"
    )]
    InvalidSimplexArity {
        /// Zero-based simplex specification index.
        simplex_index: usize,
        /// Number of supplied vertex indices.
        actual: usize,
        /// Required number of vertex indices (`D + 1`).
        expected: usize,
    },
    /// A simplex specification references a missing input vertex.
    #[error(
        "simplex {simplex_index} references vertex index {vertex_index}, but the vertex count is {bound}"
    )]
    IndexOutOfBounds {
        /// Zero-based simplex specification index.
        simplex_index: usize,
        /// Invalid input vertex index.
        vertex_index: usize,
        /// Number of supplied vertices.
        bound: usize,
    },
    /// A simplex specification repeats one input vertex.
    #[error("simplex {simplex_index} contains duplicate vertex index {vertex_index}")]
    DuplicateVertexInSimplex {
        /// Zero-based simplex specification index.
        simplex_index: usize,
        /// Repeated input vertex index.
        vertex_index: usize,
    },
    /// A vertex could not be inserted into the unpublished TDS workspace.
    #[error("vertex {vertex_index} could not be inserted during TDS construction: {source}")]
    VertexInsertion {
        /// Zero-based input vertex index.
        vertex_index: usize,
        /// Typed TDS insertion failure.
        #[source]
        source: Box<TdsConstructionError>,
    },
    /// Cross-simplex topology failed before simplex insertion began.
    #[error("explicit TDS topology is invalid: {source}")]
    TopologyValidation {
        /// Typed TDS topology failure.
        #[source]
        source: Box<TdsConstructionError>,
    },
    /// A validated simplex specification could not create a simplex value.
    #[error("simplex {simplex_index} could not be created during TDS construction: {source}")]
    SimplexCreation {
        /// Zero-based simplex specification index.
        simplex_index: usize,
        /// Typed simplex construction failure.
        #[source]
        source: SimplexValidationError,
    },
    /// A simplex could not be inserted into the unpublished TDS workspace.
    #[error("simplex {simplex_index} could not be inserted during TDS construction: {source}")]
    SimplexInsertion {
        /// Zero-based simplex specification index.
        simplex_index: usize,
        /// Typed TDS insertion failure.
        #[source]
        source: Box<TdsConstructionError>,
    },
    /// Neighbor derivation failed for the assembled connectivity.
    #[error("neighbor assignment failed during TDS construction: {source}")]
    NeighborAssignment {
        /// Typed TDS structural failure.
        #[source]
        source: Box<TdsError>,
    },
    /// Vertex-incidence derivation failed for the assembled connectivity.
    #[error("incident-simplex assignment failed during TDS construction: {source}")]
    IncidentAssignment {
        /// Typed TDS mutation failure.
        #[source]
        source: Box<TdsMutationError>,
    },
    /// The simplex complex could not be assigned coherent combinatorial orientation.
    #[error("orientation normalization failed during TDS construction: {source}")]
    OrientationNormalization {
        /// Typed TDS orientation failure.
        #[source]
        source: Box<TdsError>,
    },
    /// Final cumulative Levels 1–2 validation failed.
    #[error("Levels 1-2 validation failed during TDS construction: {source}")]
    Validation {
        /// Typed TDS validation failure.
        #[source]
        source: Box<TdsError>,
    },
}

/// Fluent builder for a proof-bearing Levels 1–2 [`Tds`].
///
/// Raw simplex specifications index the supplied vertex slice. [`build`](Self::build)
/// validates the complete request, derives all TDS-owned state, normalizes
/// coherent orientation, and returns only a constructed TDS that passes
/// [`Tds::validate`]. No geometric realization or Delaunay property is inferred
/// at this layer.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CoordinateConversionError;
/// use delaunay::prelude::tds::{TdsBuilder, TdsBuilderError};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Build(#[from] TdsBuilderError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = [
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let simplices = [vec![0, 1, 2]];
///
/// let tds = TdsBuilder::new(&vertices, &simplices).build()?;
/// assert_eq!(tds.number_of_simplices(), 1);
/// assert!(tds.validate().is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TdsBuilder<'a, U, const D: usize, V = ()> {
    input: TdsBuilderInput<'a, U, D>,
    _simplex_data: PhantomData<V>,
}

impl<'a, U, const D: usize> TdsBuilder<'a, U, D> {
    /// Creates an inert explicit-connectivity construction request.
    ///
    /// Validation is deliberately deferred to [`build`](Self::build), so
    /// creating a builder is infallible even when the raw simplex specifications
    /// are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::tds::TdsBuilder;
    ///
    /// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let simplices = [vec![0, 1, 2]];
    /// let builder = TdsBuilder::new(&vertices, &simplices);
    /// assert_eq!(builder.vertex_count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn new(vertices: &'a [Vertex<U, D>], simplices: &'a [Vec<usize>]) -> Self {
        Self {
            input: TdsBuilderInput::Raw {
                vertices,
                simplices,
            },
            _simplex_data: PhantomData,
        }
    }

    /// Creates a builder from input already parsed by a higher boundary.
    pub(crate) const fn from_parsed(input: ParsedTdsInput<'a, U, D>) -> Self {
        Self {
            input: TdsBuilderInput::Parsed(input),
            _simplex_data: PhantomData,
        }
    }
}

impl<'a, U, V, const D: usize> TdsBuilder<'a, U, D, V> {
    /// Returns the number of raw input vertices in this request.
    #[must_use]
    pub const fn vertex_count(&self) -> usize {
        self.input.vertices().len()
    }

    /// Returns the number of raw maximal-simplex specifications in this request.
    #[must_use]
    pub const fn simplex_count(&self) -> usize {
        self.input.simplices().len()
    }

    /// Selects the persisted simplex payload type without changing connectivity.
    ///
    /// The builder initializes simplex payloads as `None`; callers can populate
    /// them through checked owner methods after construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::CoordinateConversionError;
    /// use delaunay::prelude::tds::{Tds, TdsBuilder, TdsBuilderError};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Build(#[from] TdsBuilderError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [
    ///     delaunay::vertex![0.0, 0.0]?,
    ///     delaunay::vertex![1.0, 0.0]?,
    ///     delaunay::vertex![0.0, 1.0]?,
    /// ];
    /// let simplices = [vec![0, 1, 2]];
    /// let tds: Tds<(), usize, 2> = TdsBuilder::new(&vertices, &simplices)
    ///     .simplex_data_type::<usize>()
    ///     .build()?;
    /// assert_eq!(tds.number_of_simplices(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn simplex_data_type<W>(self) -> TdsBuilder<'a, U, D, W> {
        TdsBuilder {
            input: self.input,
            _simplex_data: PhantomData,
        }
    }

    /// Validates and assembles a proof-bearing Levels 1–2 TDS.
    ///
    /// The returned value has complete UUID mappings, bounded facet sharing,
    /// reciprocal neighbors, complete incidence, no duplicate simplices, and
    /// coherent combinatorial orientation. The construction state is marked
    /// complete only after [`Tds::validate`] succeeds.
    /// Input vertices and their payloads are cloned into the new TDS, so `U`
    /// need only implement [`Clone`]. The simplex payload type `V` is unbounded
    /// because explicit construction initializes each simplex payload as `None`.
    ///
    /// # Errors
    ///
    /// Returns [`TdsBuilderError`] when raw indices are malformed, element
    /// insertion fails, derived adjacency or incidence cannot be established,
    /// coherent orientation cannot be normalized, or final Levels 1–2
    /// validation fails.
    pub fn build(self) -> Result<Tds<U, V, D>, TdsBuilderError>
    where
        U: Clone,
    {
        let parsed = self.input.parse().map_err(TdsBuilderError::from)?;
        let vertices = parsed.vertices;
        let simplices = parsed.simplices;

        let mut draft = TdsDraft::new();
        let mut index_to_key = Vec::with_capacity(vertices.len());
        for (vertex_index, vertex) in vertices.iter().cloned().enumerate() {
            let vertex_key =
                draft
                    .insert_vertex(vertex)
                    .map_err(|source| TdsBuilderError::VertexInsertion {
                        vertex_index,
                        source: Box::new(source),
                    })?;
            index_to_key.push(vertex_key);
        }

        Self::validate_topology(simplices, &index_to_key).map_err(|source| {
            TdsBuilderError::TopologyValidation {
                source: Box::new(source),
            }
        })?;

        for (simplex_index, simplex_spec) in simplices.iter().enumerate() {
            let vertex_keys = Self::simplex_vertex_keys(simplex_spec, &index_to_key);
            let simplex = Simplex::try_new(vertex_keys).map_err(|source| {
                TdsBuilderError::SimplexCreation {
                    simplex_index,
                    source,
                }
            })?;
            draft
                .insert_simplex_prechecked_topology(simplex)
                .map_err(|source| TdsBuilderError::SimplexInsertion {
                    simplex_index,
                    source: Box::new(source),
                })?;
        }

        draft.finish().map_err(|source| match source {
            TdsDraftError::NeighborAssignment { source } => {
                TdsBuilderError::NeighborAssignment { source }
            }
            TdsDraftError::IncidentAssignment { source } => {
                TdsBuilderError::IncidentAssignment { source }
            }
            TdsDraftError::OrientationNormalization { source } => {
                TdsBuilderError::OrientationNormalization { source }
            }
            TdsDraftError::Validation { source } => TdsBuilderError::Validation { source },
        })
    }

    /// Proves cross-simplex uniqueness and facet multiplicity in one linear pass.
    fn validate_topology(
        simplices: &[Vec<usize>],
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        Self::reject_duplicate_simplices(simplices, index_to_key)?;
        Self::reject_overshared_facets(simplices, index_to_key)?;
        Ok(())
    }

    /// Rejects repeated maximal simplices before the unchecked bulk insertion loop.
    fn reject_duplicate_simplices(
        simplices: &[Vec<usize>],
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        let mut seen: FastHashMap<SimplexVertexKeyBuffer, usize> =
            fast_hash_map_with_capacity(simplices.len());

        for (simplex_index, simplex_spec) in simplices.iter().enumerate() {
            let mut identity = Self::simplex_vertex_keys(simplex_spec, index_to_key);
            identity.as_mut_slice().sort_unstable();
            match seen.entry(identity) {
                Entry::Occupied(entry) => {
                    let mut vertex_indices = simplex_spec.clone();
                    vertex_indices.sort_unstable();
                    return Err(TdsConstructionError::ValidationError {
                        source: TdsError::DuplicateExplicitSimplices {
                            existing_simplex_index: *entry.get(),
                            duplicate_simplex_index: simplex_index,
                            vertex_indices,
                        },
                    });
                }
                Entry::Vacant(entry) => {
                    entry.insert(simplex_index);
                }
            }
        }

        Ok(())
    }

    /// Rejects facet multiplicity above two before neighbor derivation begins.
    fn reject_overshared_facets(
        simplices: &[Vec<usize>],
        index_to_key: &[VertexKey],
    ) -> Result<(), TdsConstructionError> {
        let capacity = simplices.len().saturating_mul(D.saturating_add(1));
        let mut incident_counts: FastHashMap<SimplexVertexKeyBuffer, usize> =
            fast_hash_map_with_capacity(capacity);

        for (simplex_index, simplex_spec) in simplices.iter().enumerate() {
            for facet_index in 0..=D {
                let mut facet_identity: SimplexVertexKeyBuffer = simplex_spec
                    .iter()
                    .enumerate()
                    .filter_map(|(local_index, &input_index)| {
                        (local_index != facet_index).then_some(index_to_key[input_index])
                    })
                    .collect();
                facet_identity.as_mut_slice().sort_unstable();

                match incident_counts.entry(facet_identity) {
                    Entry::Occupied(mut entry) => {
                        let incident_count = *entry.get();
                        if incident_count >= 2 {
                            let mut facet_vertex_indices: Vec<usize> = simplex_spec
                                .iter()
                                .enumerate()
                                .filter_map(|(local_index, &input_index)| {
                                    (local_index != facet_index).then_some(input_index)
                                })
                                .collect();
                            facet_vertex_indices.sort_unstable();
                            return Err(TdsConstructionError::ValidationError {
                                source: TdsError::ExplicitFacetSharingViolation {
                                    facet_key: facet_key_from_vertices(entry.key().as_slice()),
                                    facet_vertex_indices,
                                    existing_incident_count: incident_count,
                                    attempted_incident_count: incident_count + 1,
                                    max_incident_count: 2,
                                    candidate_simplex_index: simplex_index,
                                    candidate_facet_index: facet_index,
                                },
                            });
                        }
                        *entry.get_mut() += 1;
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(1);
                    }
                }
            }
        }

        Ok(())
    }

    /// Maps prevalidated input indices into canonical TDS vertex keys.
    fn simplex_vertex_keys(
        simplex_spec: &[usize],
        index_to_key: &[VertexKey],
    ) -> SimplexVertexKeyBuffer {
        simplex_spec
            .iter()
            .map(|&vertex_index| index_to_key[vertex_index])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TriangulationConstructionState;
    use crate::vertex;
    use std::assert_matches;

    /// Supplies a two-simplex disk whose raw orientations require normalization.
    fn two_triangle_fixture() -> ([Vertex<(), 2>; 4], [Vec<usize>; 2]) {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
            vertex![1.0, 0.2].unwrap(),
        ];
        let simplices = [vec![0, 1, 3], vec![0, 2, 3]];
        (vertices, simplices)
    }

    #[test]
    fn build_publishes_only_a_complete_valid_tds() {
        let (vertices, simplices) = two_triangle_fixture();
        let builder = TdsBuilder::new(&vertices, &simplices).simplex_data_type::<usize>();
        assert_eq!(builder.vertex_count(), 4);
        assert_eq!(builder.simplex_count(), 2);

        let tds = builder.build().unwrap();

        assert_matches!(
            tds.construction_state(),
            TriangulationConstructionState::Constructed
        );
        assert!(tds.validate().is_ok());
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_simplices(), 2);
        assert!(tds.simplices().all(|(_, simplex)| simplex.data().is_none()));
    }

    #[test]
    fn build_rejects_raw_specs_before_assembly() {
        let vertices = [
            vertex![0.0, 0.0].unwrap(),
            vertex![1.0, 0.0].unwrap(),
            vertex![0.0, 1.0].unwrap(),
        ];

        assert_matches!(
            TdsBuilder::new(&vertices, &[]).build(),
            Err(TdsBuilderError::EmptySimplices)
        );
        assert_matches!(
            TdsBuilder::new(&vertices, &[vec![0, 1]]).build(),
            Err(TdsBuilderError::InvalidSimplexArity {
                simplex_index: 0,
                actual: 2,
                expected: 3,
            })
        );
        assert_matches!(
            TdsBuilder::new(&vertices, &[vec![0, 1, 3]]).build(),
            Err(TdsBuilderError::IndexOutOfBounds {
                simplex_index: 0,
                vertex_index: 3,
                bound: 3,
            })
        );
        assert_matches!(
            TdsBuilder::new(&vertices, &[vec![0, 1, 1]]).build(),
            Err(TdsBuilderError::DuplicateVertexInSimplex {
                simplex_index: 0,
                vertex_index: 1,
            })
        );
    }

    #[test]
    fn build_reports_the_input_index_for_a_duplicate_vertex_uuid() {
        let repeated = vertex![0.0, 0.0].unwrap();
        let repeated_uuid = repeated.uuid();
        let vertices = [repeated, repeated, vertex![0.0, 1.0].unwrap()];
        let simplices = [vec![0, 1, 2]];

        assert_matches!(
            TdsBuilder::new(&vertices, &simplices).build(),
            Err(TdsBuilderError::VertexInsertion {
                vertex_index: 1,
                source,
            }) if matches!(
                source.as_ref(),
                TdsConstructionError::DuplicateUuid {
                    entity: crate::core::tds::EntityKind::Vertex,
                    uuid,
                } if *uuid == repeated_uuid
            )
        );
    }

    #[test]
    fn build_accepts_the_vacuously_valid_empty_complex() {
        let vertices: [Vertex<(), 2>; 0] = [];
        let simplices: [Vec<usize>; 0] = [];

        let tds = TdsBuilder::new(&vertices, &simplices).build().unwrap();
        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_simplices(), 0);
        assert!(tds.validate().is_ok());
    }

    #[test]
    fn parsed_input_flows_into_assembly_without_returning_to_raw_state() {
        let (vertices, simplices) = two_triangle_fixture();
        let parsed = ParsedTdsInput::try_new(&vertices, &simplices).unwrap();

        let tds = TdsBuilder::from_parsed(parsed).build().unwrap();

        assert_eq!(tds.number_of_vertices(), vertices.len());
        assert_eq!(tds.number_of_simplices(), simplices.len());
        assert!(tds.validate().is_ok());
    }
}
