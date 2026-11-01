//! Generic simplicial-complex export data for notebooks and downstream tools.
//!
//! This module owns a downstream-facing schema distinct from the internal TDS
//! persistence format. It uses stable vertex and simplex UUIDs for entity
//! identity so consumers do not depend on storage-local slotmap keys.

#![forbid(unsafe_code)]

use crate::core::collections::NeighborBuffer;
use crate::core::simplex::NeighborSlot;
use crate::core::tds::{SimplexKey, VertexKey};
use crate::core::vertex::Vertex;
use crate::delaunay_model::DelaunayTriangulation;
use crate::geometry::traits::coordinate::InvalidCoordinateValue;
use crate::topology::traits::topological_space::{GlobalTopology, TopologyKind};
use crate::triangulation::Triangulation;
use crate::triangulation::validation::TopologyGuarantee;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;
use uuid::Uuid;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
};

/// Supplies absent optional attributes during deserialization without requiring
/// the attribute payload type to implement `Default`.
const fn no_attributes<Attributes>() -> Option<Attributes> {
    None
}

/// Schema name used by [`VisualizationData`].
pub const VISUALIZATION_SCHEMA: &str = "delaunay.simplicial_complex";

/// Current [`VisualizationData`] schema version.
pub const VISUALIZATION_SCHEMA_VERSION: u32 = 2;

/// Stable topology-kind schema category used by [`VisualizationMetadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VisualizationTopologyKind {
    /// Euclidean topology.
    Euclidean,
    /// Toroidal topology.
    Toroidal,
    /// Spherical topology.
    Spherical,
    /// Hyperbolic topology.
    Hyperbolic,
    /// Unknown topology-kind text from an external JSON producer.
    Unknown {
        /// Raw schema text that did not match a known topology kind.
        actual: String,
    },
}

impl VisualizationTopologyKind {
    /// Returns the JSON schema spelling for this topology-kind category.
    fn schema_name(&self) -> &str {
        match self {
            Self::Euclidean => "Euclidean",
            Self::Toroidal => "Toroidal",
            Self::Spherical => "Spherical",
            Self::Hyperbolic => "Hyperbolic",
            Self::Unknown { actual } => actual,
        }
    }

    /// Reports whether this topology-kind value is part of the schema.
    const fn is_supported(&self) -> bool {
        match self {
            Self::Euclidean | Self::Toroidal | Self::Spherical | Self::Hyperbolic => true,
            Self::Unknown { .. } => false,
        }
    }
}

impl From<TopologyKind> for VisualizationTopologyKind {
    fn from(kind: TopologyKind) -> Self {
        match kind {
            TopologyKind::Euclidean => Self::Euclidean,
            TopologyKind::Toroidal => Self::Toroidal,
            TopologyKind::Spherical => Self::Spherical,
            TopologyKind::Hyperbolic => Self::Hyperbolic,
        }
    }
}

impl fmt::Display for VisualizationTopologyKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.schema_name())
    }
}

impl Serialize for VisualizationTopologyKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.schema_name())
    }
}

impl<'de> Deserialize<'de> for VisualizationTopologyKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let actual = String::deserialize(deserializer)?;
        Ok(match actual.as_str() {
            "Euclidean" => Self::Euclidean,
            "Toroidal" => Self::Toroidal,
            "Spherical" => Self::Spherical,
            "Hyperbolic" => Self::Hyperbolic,
            _ => Self::Unknown { actual },
        })
    }
}

/// Stable topology-guarantee schema category used by [`VisualizationMetadata`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VisualizationTopologyGuarantee {
    /// Pseudomanifold guarantee.
    Pseudomanifold,
    /// PL-manifold guarantee.
    PLManifold,
    /// Unknown topology-guarantee text from an external JSON producer.
    Unknown {
        /// Raw schema text that did not match a known topology guarantee.
        actual: String,
    },
}

impl VisualizationTopologyGuarantee {
    /// Returns the JSON schema spelling for this topology-guarantee category.
    fn schema_name(&self) -> &str {
        match self {
            Self::Pseudomanifold => "Pseudomanifold",
            Self::PLManifold => "PLManifold",
            Self::Unknown { actual } => actual,
        }
    }

    /// Reports whether this topology-guarantee value is part of the schema.
    const fn is_supported(&self) -> bool {
        match self {
            Self::Pseudomanifold | Self::PLManifold => true,
            Self::Unknown { .. } => false,
        }
    }
}

impl From<TopologyGuarantee> for VisualizationTopologyGuarantee {
    fn from(guarantee: TopologyGuarantee) -> Self {
        match guarantee {
            TopologyGuarantee::Pseudomanifold => Self::Pseudomanifold,
            TopologyGuarantee::PLManifold => Self::PLManifold,
        }
    }
}

impl fmt::Display for VisualizationTopologyGuarantee {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.schema_name())
    }
}

impl Serialize for VisualizationTopologyGuarantee {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.schema_name())
    }
}

impl<'de> Deserialize<'de> for VisualizationTopologyGuarantee {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let actual = String::deserialize(deserializer)?;
        Ok(match actual.as_str() {
            "Pseudomanifold" => Self::Pseudomanifold,
            "PLManifold" => Self::PLManifold,
            _ => Self::Unknown { actual },
        })
    }
}

/// Generic simplicial-complex data for analysis, notebooks, and interchange.
///
/// The JSON shape is intentionally separate from `Tds` serde persistence. It
/// exposes common visualization primitives: schema metadata, stable vertex and
/// simplex ids, coordinates, simplex vertex membership, and facet-neighbor
/// adjacency by stable simplex id. Attribute type parameters let downstream
/// crates wrap or extend the base records with domain-specific metadata.
/// Toroidal snapshots also retain periods and ordered lattice offsets in
/// [`Self::periodic`], allowing repeated quotient UUIDs to identify distinct
/// lifted vertex images. Schema v2 adds this realization data; nonperiodic v1
/// records remain accepted by validation.
/// The value is an owned, detached interchange snapshot rather than a live
/// borrowed view over a triangulation; regenerate it after mutating the source
/// triangulation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::construction::{
///     DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
/// };
/// use delaunay::prelude::export::VisualizationExportError;
/// use delaunay::prelude::geometry::CoordinateConversionError;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Export(#[from] VisualizationExportError),
/// #     #[error(transparent)]
/// #     Serde(#[from] serde_json::Error),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     vertex![0.0, 0.0]?,
///     vertex![1.0, 0.0]?,
///     vertex![0.0, 1.0]?,
/// ];
/// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// let export = triangulation.to_visualization_data()?;
/// let json = serde_json::to_string(&export)?;
///
/// assert!(json.contains("\"schema\":\"delaunay.simplicial_complex\""));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VisualizationData<
    const D: usize,
    VertexAttributes = (),
    SimplexAttributes = (),
    AdjacencyAttributes = (),
    GlobalAttributes = (),
> {
    /// Global schema and producer metadata.
    pub metadata: VisualizationMetadata<GlobalAttributes>,
    /// Vertices sorted by stable UUID for deterministic output.
    pub vertices: Vec<VertexRecord<D, VertexAttributes>>,
    /// Maximal simplices sorted by stable UUID for deterministic output.
    pub simplices: Vec<SimplexRecord<SimplexAttributes>>,
    /// Facet-neighbor adjacency records sorted by simplex id and facet index.
    pub adjacency: Vec<AdjacencyRecord<AdjacencyAttributes>>,
    /// Toroidal periods and lifted vertex offsets, required for toroidal data.
    ///
    /// Absent for other topology kinds and for legacy nonperiodic v1 records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub periodic: Option<PeriodicVisualizationData>,
}

/// Periodic realization data for a toroidal [`VisualizationData`] snapshot.
///
/// A simplex vertex at slot `i` is realized at
/// `coordinates + simplex_vertex_offsets[simplex_id][i] * periods`, component
/// by component. Repeated quotient vertex UUIDs therefore remain distinct when
/// their lattice offsets differ. Facets match up to a common lattice translation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PeriodicVisualizationData {
    /// Positive finite fundamental-domain lengths in dimension order.
    pub periods: Vec<f64>,
    /// Ordered lattice offsets for every simplex, keyed by its stable UUID.
    pub simplex_vertex_offsets: BTreeMap<Uuid, Vec<Vec<i8>>>,
}

impl<const D: usize, VertexAttributes, SimplexAttributes, AdjacencyAttributes, GlobalAttributes>
    VisualizationData<D, VertexAttributes, SimplexAttributes, AdjacencyAttributes, GlobalAttributes>
{
    /// Parses this raw visualization/interchange value into a validated wrapper.
    ///
    /// This consumes the raw DTO and returns a proof-bearing value whose
    /// borrowed accessors can be used without repeating schema validation.
    ///
    /// # Errors
    ///
    /// Returns [`VisualizationDataValidationError`] when schema metadata,
    /// coordinate values or arity, non-nil ids, connectivity, facet coverage,
    /// or reciprocal facet-compatible adjacency, including distinct reciprocal
    /// records for self-neighbor facets, are inconsistent.
    pub fn into_validated(
        self,
    ) -> Result<
        ValidatedVisualizationData<
            D,
            VertexAttributes,
            SimplexAttributes,
            AdjacencyAttributes,
            GlobalAttributes,
        >,
        VisualizationDataValidationError,
    > {
        ValidatedVisualizationData::try_from_raw(self)
    }

    /// Validates this raw visualization/interchange value against the schema.
    ///
    /// This is the boundary to use after deserializing untrusted JSON. Produced
    /// values from [`DelaunayTriangulation::to_visualization_data`] should
    /// already satisfy these invariants. Validation does not convert the value
    /// into canonical `Tds` storage; use [`Self::into_validated`] when callers
    /// need to carry validation evidence inward, or use triangulation/TDS serde
    /// for validated Rust hydration.
    ///
    /// # Errors
    ///
    /// Returns [`VisualizationDataValidationError`] when schema metadata,
    /// coordinate values or arity, non-nil ids, connectivity, facet coverage,
    /// or reciprocal facet-compatible adjacency, including distinct reciprocal
    /// records for self-neighbor facets, are inconsistent.
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
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let export = triangulation.to_visualization_data()?;
    ///
    /// export.validate()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate(&self) -> Result<(), VisualizationDataValidationError> {
        let expected_simplex_vertices = visualization_arity::<D>()?;
        validate_metadata::<D, GlobalAttributes>(
            &self.metadata,
            self.vertices.len(),
            self.simplices.len(),
        )?;
        validate_periodic_metadata::<D, _, _>(
            &self.metadata,
            self.periodic.as_ref(),
            &self.simplices,
            expected_simplex_vertices,
        )?;

        let mut vertex_ids = HashSet::with_capacity(self.vertices.len());
        for (vertex_index, vertex) in self.vertices.iter().enumerate() {
            if vertex.id.is_nil() {
                return Err(VisualizationDataValidationError::NilVertexId { vertex_index });
            }
            if vertex.coordinates.len() != D {
                return Err(
                    VisualizationDataValidationError::InvalidVertexCoordinateCount {
                        vertex_id: vertex.id,
                        expected: D,
                        actual: vertex.coordinates.len(),
                    },
                );
            }
            for (coordinate_index, coordinate) in vertex.coordinates.iter().enumerate() {
                if !coordinate.is_finite() {
                    return Err(
                        VisualizationDataValidationError::InvalidVertexCoordinateValue {
                            vertex_id: vertex.id,
                            coordinate_index,
                            value: InvalidCoordinateValue::from_debug(coordinate),
                        },
                    );
                }
            }
            if !vertex_ids.insert(vertex.id) {
                return Err(VisualizationDataValidationError::DuplicateVertexId {
                    vertex_id: vertex.id,
                });
            }
        }

        let mut simplex_ids = HashSet::with_capacity(self.simplices.len());
        for (simplex_index, simplex) in self.simplices.iter().enumerate() {
            if simplex.id.is_nil() {
                return Err(VisualizationDataValidationError::NilSimplexId { simplex_index });
            }
            if !simplex_ids.insert(simplex.id) {
                return Err(VisualizationDataValidationError::DuplicateSimplexId {
                    simplex_id: simplex.id,
                });
            }
            if simplex.vertex_ids.len() != expected_simplex_vertices {
                return Err(
                    VisualizationDataValidationError::InvalidSimplexVertexCount {
                        simplex_id: simplex.id,
                        expected: expected_simplex_vertices,
                        actual: simplex.vertex_ids.len(),
                    },
                );
            }

            let mut local_vertex_ids = HashSet::with_capacity(simplex.vertex_ids.len());
            let offsets = self
                .periodic
                .as_ref()
                .and_then(|periodic| periodic.simplex_vertex_offsets.get(&simplex.id));
            for (vertex_index, vertex_id) in simplex.vertex_ids.iter().enumerate() {
                if !vertex_ids.contains(vertex_id) {
                    return Err(VisualizationDataValidationError::MissingSimplexVertex {
                        simplex_id: simplex.id,
                        vertex_id: *vertex_id,
                    });
                }
                let offset = offsets.map(|offsets| offsets[vertex_index].as_slice());
                if !local_vertex_ids.insert((*vertex_id, offset)) {
                    return Err(VisualizationDataValidationError::DuplicateSimplexVertex {
                        simplex_id: simplex.id,
                        vertex_id: *vertex_id,
                    });
                }
            }
        }

        validate_adjacency::<D, _, _>(&self.adjacency, &self.simplices, self.periodic.as_ref())
    }
}

/// Proof-bearing wrapper for a validated [`VisualizationData`] value.
///
/// Raw visualization data keeps public fields for serde, external JSON, and
/// fixtures. This wrapper is the parsed form to pass inward once schema
/// metadata, finite coordinate values, dimensions, non-nil ids, connectivity,
/// facet coverage, and reciprocal facet-compatible adjacency have been checked.
/// Self-neighbor facets require distinct reciprocal records so a one-sided
/// self-gluing cannot validate as closed topology. Parsing canonicalizes record
/// order by stable ids so serialized validated values remain deterministic.
/// This remains an owned, detached snapshot; its accessors borrow records from
/// the validated snapshot, not from the source triangulation.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ValidatedVisualizationData<
    const D: usize,
    VertexAttributes = (),
    SimplexAttributes = (),
    AdjacencyAttributes = (),
    GlobalAttributes = (),
> {
    inner: VisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    >,
}

impl<const D: usize, VertexAttributes, SimplexAttributes, AdjacencyAttributes, GlobalAttributes>
    ValidatedVisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    >
{
    /// Parses a raw visualization/interchange DTO into a validated wrapper.
    /// Valid inputs are canonicalized by stable ids before storage.
    ///
    /// # Errors
    ///
    /// Returns [`VisualizationDataValidationError`] when schema metadata,
    /// coordinate values or arity, non-nil ids, connectivity, facet coverage,
    /// or reciprocal facet-compatible adjacency, including distinct reciprocal
    /// records for self-neighbor facets, are inconsistent.
    pub fn try_from_raw(
        mut raw: VisualizationData<
            D,
            VertexAttributes,
            SimplexAttributes,
            AdjacencyAttributes,
            GlobalAttributes,
        >,
    ) -> Result<Self, VisualizationDataValidationError> {
        raw.validate()?;
        canonicalize_record_order(&mut raw);
        Ok(Self { inner: raw })
    }

    /// Returns the validated raw DTO without exposing mutable access.
    pub const fn as_raw(
        &self,
    ) -> &VisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    > {
        &self.inner
    }

    /// Consumes the wrapper and returns the raw DTO, discarding validation proof.
    pub fn into_raw(
        self,
    ) -> VisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    > {
        self.inner
    }

    /// Returns validated schema and producer metadata.
    pub const fn metadata(&self) -> &VisualizationMetadata<GlobalAttributes> {
        &self.inner.metadata
    }

    /// Returns validated vertex records.
    pub fn vertices(&self) -> &[VertexRecord<D, VertexAttributes>] {
        &self.inner.vertices
    }

    /// Returns validated simplex records.
    pub fn simplices(&self) -> &[SimplexRecord<SimplexAttributes>] {
        &self.inner.simplices
    }

    /// Returns validated adjacency records.
    pub fn adjacency(&self) -> &[AdjacencyRecord<AdjacencyAttributes>] {
        &self.inner.adjacency
    }
}

impl<const D: usize, VertexAttributes, SimplexAttributes, AdjacencyAttributes, GlobalAttributes>
    TryFrom<
        VisualizationData<
            D,
            VertexAttributes,
            SimplexAttributes,
            AdjacencyAttributes,
            GlobalAttributes,
        >,
    >
    for ValidatedVisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    >
{
    type Error = VisualizationDataValidationError;

    fn try_from(
        raw: VisualizationData<
            D,
            VertexAttributes,
            SimplexAttributes,
            AdjacencyAttributes,
            GlobalAttributes,
        >,
    ) -> Result<Self, Self::Error> {
        Self::try_from_raw(raw)
    }
}

/// Global metadata for [`VisualizationData`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct VisualizationMetadata<Attributes = ()> {
    /// Stable schema name for downstream format dispatch.
    pub schema: String,
    /// Integer schema version for compatibility checks.
    pub schema_version: u32,
    /// Name of the crate or tool that produced this export.
    pub producer: String,
    /// Compile-time triangulation dimension recorded for non-Rust consumers.
    pub dimension: usize,
    /// Number of exported vertex records.
    pub vertex_count: usize,
    /// Number of exported simplex records.
    pub simplex_count: usize,
    /// Topological-space kind recorded as a stable schema category.
    pub topology_kind: VisualizationTopologyKind,
    /// Validation/topology guarantee recorded as a stable schema category.
    pub topology_guarantee: VisualizationTopologyGuarantee,
    /// Optional downstream global metadata.
    #[serde(default = "no_attributes", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attributes>,
}

/// Vertex record in [`VisualizationData`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VertexRecord<const D: usize, Attributes = ()> {
    /// Stable vertex id.
    pub id: Uuid,
    /// Vertex coordinates in triangulation dimension order.
    pub coordinates: Vec<f64>,
    /// Optional downstream per-vertex metadata.
    #[serde(default = "no_attributes", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attributes>,
}

/// Simplex record in [`VisualizationData`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SimplexRecord<Attributes = ()> {
    /// Stable simplex id.
    pub id: Uuid,
    /// Stable vertex ids in the simplex's stored orientation/order.
    pub vertex_ids: Vec<Uuid>,
    /// Optional downstream per-simplex metadata.
    #[serde(default = "no_attributes", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attributes>,
}

/// Facet-neighbor adjacency record in [`VisualizationData`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AdjacencyRecord<Attributes = ()> {
    /// Stable simplex id for the source simplex.
    pub simplex_id: Uuid,
    /// Local facet index, opposite `SimplexRecord::vertex_ids[facet_index]`.
    pub facet_index: usize,
    /// Stable id of the neighboring simplex across this facet.
    ///
    /// `None` denotes a boundary facet. When this points back to `simplex_id`,
    /// validation still requires a distinct reciprocal adjacency record for the
    /// matching self-neighbor facet.
    pub neighbor_simplex_id: Option<Uuid>,
    /// Optional downstream per-adjacency metadata.
    #[serde(default = "no_attributes", skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Attributes>,
}

/// Errors that can occur while exporting visualization data.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum VisualizationExportError {
    /// The const-generic dimension cannot represent simplex arity.
    #[error("visualization dimension {dimension} cannot represent simplex arity")]
    DimensionOutOfRange {
        /// Unsupported const-generic dimension.
        dimension: usize,
    },
    /// A simplex referenced a vertex key that is not present in the TDS.
    #[error("simplex {simplex_id} references missing vertex key {vertex_key:?}")]
    MissingVertex {
        /// Stable id of the simplex containing the bad reference.
        simplex_id: Uuid,
        /// Storage-local key that could not be resolved.
        vertex_key: VertexKey,
    },
    /// A simplex has no assigned neighbor buffer.
    #[error("simplex {simplex_id} has no assigned neighbor buffer")]
    UnassignedNeighborBuffer {
        /// Stable id of the simplex with no neighbor buffer.
        simplex_id: Uuid,
    },
    /// A simplex neighbor buffer length does not match the simplex dimension.
    #[error("simplex {simplex_id} has {actual} neighbor slots; expected {expected}")]
    InvalidNeighborCount {
        /// Stable id of the simplex with malformed neighbor slots.
        simplex_id: Uuid,
        /// Expected number of neighbor slots.
        expected: usize,
        /// Actual number of neighbor slots.
        actual: usize,
    },
    /// A simplex has an explicit unassigned neighbor slot.
    #[error("simplex {simplex_id} has an unassigned neighbor slot at facet {facet_index}")]
    UnassignedNeighborSlot {
        /// Stable id of the simplex with an unassigned slot.
        simplex_id: Uuid,
        /// Facet index containing the unassigned slot.
        facet_index: usize,
    },
    /// A simplex neighbor key does not resolve to a simplex.
    #[error(
        "simplex {simplex_id} facet {facet_index} references missing neighbor key {neighbor_key:?}"
    )]
    MissingNeighbor {
        /// Stable id of the simplex containing the bad neighbor reference.
        simplex_id: Uuid,
        /// Facet index containing the neighbor key.
        facet_index: usize,
        /// Storage-local neighbor key that could not be resolved.
        neighbor_key: SimplexKey,
    },
}

/// Errors found when validating deserialized visualization data.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum VisualizationDataValidationError {
    /// The const-generic dimension cannot represent simplex arity.
    #[error("visualization dimension {dimension} cannot represent simplex arity")]
    DimensionOutOfRange {
        /// Unsupported const-generic dimension.
        dimension: usize,
    },
    /// Toroidal data requires the periodic realization fields introduced in v2.
    #[error("toroidal visualization requires schema v2 and periodic realization data")]
    MissingPeriodicData {},
    /// Periodic realization data appeared outside the v2 toroidal schema.
    #[error("periodic realization data requires schema v2 and toroidal topology")]
    UnexpectedPeriodicData {},
    /// The periodic domain has the wrong number of periods.
    #[error("periodic domain has {actual} periods; expected {expected}")]
    InvalidPeriodCount {
        /// Required number of periods.
        expected: usize,
        /// Number of supplied periods.
        actual: usize,
    },
    /// A fundamental-domain period is not finite and strictly positive.
    #[error("periodic domain period {axis} is not finite and strictly positive")]
    InvalidPeriod {
        /// Axis with the invalid period.
        axis: usize,
    },
    /// Periodic offsets do not cover a simplex in the simplex table.
    #[error("periodic offsets are missing for simplex {simplex_id}")]
    MissingPeriodicOffsets {
        /// Simplex missing its realization data.
        simplex_id: Uuid,
    },
    /// Periodic offsets refer to a simplex absent from the simplex table.
    #[error("periodic offsets reference missing simplex {simplex_id}")]
    UnknownPeriodicSimplex {
        /// Simplex absent from the simplex table.
        simplex_id: Uuid,
    },
    /// A simplex has the wrong number of periodic vertex offsets.
    #[error("simplex {simplex_id} has {actual} periodic offsets; expected {expected}")]
    InvalidPeriodicOffsetCount {
        /// Simplex with malformed offset data.
        simplex_id: Uuid,
        /// Required offset count.
        expected: usize,
        /// Supplied offset count.
        actual: usize,
    },
    /// A periodic lattice offset has the wrong coordinate arity.
    #[error(
        "simplex {simplex_id} vertex {vertex_index} offset has {actual} entries; expected {expected}"
    )]
    InvalidPeriodicOffsetDimension {
        /// Simplex with malformed offset data.
        simplex_id: Uuid,
        /// Vertex slot with the malformed offset.
        vertex_index: usize,
        /// Required coordinate count.
        expected: usize,
        /// Supplied coordinate count.
        actual: usize,
    },
    /// Reciprocal periodic facets do not agree up to a common lattice translation.
    #[error(
        "simplex {simplex_id} facet {facet_index} has no matching reciprocal periodic facet in {neighbor_simplex_id}"
    )]
    InvalidPeriodicFacetSharing {
        /// Source simplex of the invalid adjacency.
        simplex_id: Uuid,
        /// Source facet index.
        facet_index: usize,
        /// Neighbor without a matching reciprocal facet.
        neighbor_simplex_id: Uuid,
    },
    /// The schema name is not supported by this validator.
    #[error("unsupported visualization schema {actual:?}; expected {expected:?}")]
    InvalidSchema {
        /// Expected schema name.
        expected: &'static str,
        /// Actual schema name.
        actual: String,
    },
    /// The schema version is not supported by this validator.
    #[error("unsupported visualization schema version {actual}; expected {expected}")]
    InvalidSchemaVersion {
        /// Expected schema version.
        expected: u32,
        /// Actual schema version.
        actual: u32,
    },
    /// Metadata dimension does not match the const-generic export dimension.
    #[error("metadata dimension {actual} does not match export dimension {expected}")]
    DimensionMismatch {
        /// Expected const-generic dimension.
        expected: usize,
        /// Actual metadata dimension.
        actual: usize,
    },
    /// Metadata vertex count does not match the vertex table length.
    #[error("metadata vertex_count {expected} does not match {actual} vertex records")]
    VertexCountMismatch {
        /// Metadata vertex count.
        expected: usize,
        /// Actual vertex table length.
        actual: usize,
    },
    /// Metadata simplex count does not match the simplex table length.
    #[error("metadata simplex_count {expected} does not match {actual} simplex records")]
    SimplexCountMismatch {
        /// Metadata simplex count.
        expected: usize,
        /// Actual simplex table length.
        actual: usize,
    },
    /// Metadata topology kind is not one of the schema names.
    #[error(
        "unsupported topology kind {actual}; expected one of Euclidean, Toroidal, Spherical, Hyperbolic"
    )]
    InvalidTopologyKind {
        /// Actual topology-kind schema category.
        actual: VisualizationTopologyKind,
    },
    /// Metadata topology guarantee is not one of the schema names.
    #[error("unsupported topology guarantee {actual}; expected one of Pseudomanifold, PLManifold")]
    InvalidTopologyGuarantee {
        /// Actual topology-guarantee schema category.
        actual: VisualizationTopologyGuarantee,
    },
    /// A vertex record uses the nil UUID rather than a stable entity id.
    #[error("vertex record {vertex_index} uses nil UUID")]
    NilVertexId {
        /// Position of the invalid vertex record in the raw vertex table.
        vertex_index: usize,
    },
    /// A vertex has the wrong coordinate arity.
    #[error("vertex {vertex_id} has {actual} coordinates; expected {expected}")]
    InvalidVertexCoordinateCount {
        /// Stable vertex id.
        vertex_id: Uuid,
        /// Expected coordinate count.
        expected: usize,
        /// Actual coordinate count.
        actual: usize,
    },
    /// A vertex coordinate is not finite.
    #[error("vertex {vertex_id} coordinate {coordinate_index} is non-finite: {value}")]
    InvalidVertexCoordinateValue {
        /// Stable vertex id.
        vertex_id: Uuid,
        /// Coordinate index containing the non-finite value.
        coordinate_index: usize,
        /// Non-finite coordinate category.
        value: InvalidCoordinateValue,
    },
    /// The vertex table contains a duplicate stable id.
    #[error("duplicate vertex id {vertex_id}")]
    DuplicateVertexId {
        /// Duplicated stable vertex id.
        vertex_id: Uuid,
    },
    /// A simplex record uses the nil UUID rather than a stable entity id.
    #[error("simplex record {simplex_index} uses nil UUID")]
    NilSimplexId {
        /// Position of the invalid simplex record in the raw simplex table.
        simplex_index: usize,
    },
    /// A simplex has the wrong vertex arity.
    #[error("simplex {simplex_id} has {actual} vertex ids; expected {expected}")]
    InvalidSimplexVertexCount {
        /// Stable simplex id.
        simplex_id: Uuid,
        /// Expected vertex id count.
        expected: usize,
        /// Actual vertex id count.
        actual: usize,
    },
    /// The simplex table contains a duplicate stable id.
    #[error("duplicate simplex id {simplex_id}")]
    DuplicateSimplexId {
        /// Duplicated stable simplex id.
        simplex_id: Uuid,
    },
    /// A simplex references a vertex id absent from the vertex table.
    #[error("simplex {simplex_id} references missing vertex id {vertex_id}")]
    MissingSimplexVertex {
        /// Stable simplex id.
        simplex_id: Uuid,
        /// Missing stable vertex id.
        vertex_id: Uuid,
    },
    /// A simplex repeats a vertex id (and, for toroidal data, its lattice offset).
    #[error("simplex {simplex_id} repeats vertex id {vertex_id}")]
    DuplicateSimplexVertex {
        /// Stable simplex id.
        simplex_id: Uuid,
        /// Repeated stable vertex id.
        vertex_id: Uuid,
    },
    /// An adjacency record references an absent source simplex id.
    #[error("adjacency references missing source simplex id {simplex_id}")]
    MissingAdjacencySimplex {
        /// Missing stable source simplex id.
        simplex_id: Uuid,
    },
    /// An adjacency record uses an invalid facet index.
    #[error(
        "adjacency for simplex {simplex_id} has facet index {facet_index}; expected less than {max_exclusive}"
    )]
    InvalidAdjacencyFacetIndex {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Invalid facet index.
        facet_index: usize,
        /// Exclusive upper bound for valid facet indices.
        max_exclusive: usize,
    },
    /// An adjacency record references an absent neighbor simplex id.
    #[error(
        "adjacency for simplex {simplex_id} facet {facet_index} references missing neighbor simplex id {neighbor_simplex_id}"
    )]
    MissingAdjacencyNeighbor {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Facet index for the bad neighbor reference.
        facet_index: usize,
        /// Missing stable neighbor simplex id.
        neighbor_simplex_id: Uuid,
    },
    /// A non-boundary adjacency record points at a neighbor that does not share the source facet.
    #[error(
        "adjacency for simplex {simplex_id} facet {facet_index} references neighbor {neighbor_simplex_id}, but source facet vertex {missing_vertex_id} is absent from the neighbor"
    )]
    InvalidAdjacencyFacetSharing {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Facet index for the invalid neighbor reference.
        facet_index: usize,
        /// Neighbor simplex id that does not contain the source facet.
        neighbor_simplex_id: Uuid,
        /// Source-facet vertex that the neighbor simplex does not contain.
        missing_vertex_id: Uuid,
    },
    /// The adjacency table has multiple records for one simplex facet.
    #[error("duplicate adjacency record for simplex {simplex_id} facet {facet_index}")]
    DuplicateAdjacency {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Duplicated facet index.
        facet_index: usize,
    },
    /// The adjacency table is missing a required simplex-facet record.
    #[error("missing adjacency record for simplex {simplex_id} facet {facet_index}")]
    MissingAdjacency {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Missing facet index.
        facet_index: usize,
    },
    /// A non-boundary adjacency record is not reciprocated by the neighbor simplex.
    #[error(
        "adjacency for simplex {simplex_id} facet {facet_index} references neighbor {neighbor_simplex_id}, but the neighbor does not reference it back"
    )]
    AsymmetricAdjacency {
        /// Stable source simplex id.
        simplex_id: Uuid,
        /// Facet index for the asymmetric neighbor reference.
        facet_index: usize,
        /// Neighbor simplex id that does not reciprocate the link.
        neighbor_simplex_id: Uuid,
    },
}

impl<K, U, V, const D: usize> Triangulation<K, U, V, D> {
    /// Exports this triangulation as generic visualization/interchange data.
    ///
    /// The returned value implements [`Serialize`] and [`Deserialize`] so callers
    /// can write JSON with `serde_json`, feed the schema to notebooks or ML
    /// pipelines, or adapt it for visualization and external editing tools.
    /// Entity ids are vertex/simplex UUIDs, not runtime `VertexKey` or
    /// `SimplexKey` debug strings.
    ///
    /// The returned value is an owned, detached snapshot of the triangulation's
    /// current UUIDs, coordinates, simplex membership, and facet adjacency.
    /// Toroidal exports include domain periods and vertex offsets in
    /// [`VisualizationData::periodic`] so their lifted realization is preserved.
    /// Mutating the triangulation later does not update this export; regenerate
    /// the value when consumers need a fresh view of the topology.
    ///
    /// # Errors
    ///
    /// Returns [`VisualizationExportError`] if the stored topology has missing
    /// vertex references, missing neighbor references, or unassigned neighbor
    /// slots. Valid constructed triangulations should export without error.
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
    ///     vertex![0.0, 0.0, 0.0]?,
    ///     vertex![1.0, 0.0, 0.0]?,
    ///     vertex![0.0, 1.0, 0.0]?,
    ///     vertex![0.0, 0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    ///
    /// let export = triangulation.to_visualization_data()?;
    ///
    /// assert_eq!(export.metadata.schema, "delaunay.simplicial_complex");
    /// assert_eq!(export.metadata.dimension, 3);
    /// assert_eq!(export.vertices.len(), 4);
    /// assert_eq!(export.simplices.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_visualization_data(&self) -> Result<VisualizationData<D>, VisualizationExportError> {
        let simplex_arity = D
            .checked_add(1)
            .ok_or(VisualizationExportError::DimensionOutOfRange { dimension: D })?;
        let mut periodic = match self.global_topology() {
            GlobalTopology::Toroidal { domain, .. } => Some(PeriodicVisualizationData {
                periods: domain.periods().to_vec(),
                simplex_vertex_offsets: BTreeMap::new(),
            }),
            _ => None,
        };
        let mut vertices: Vec<_> = self
            .vertices()
            .map(|(_, vertex)| VertexRecord {
                id: vertex.uuid(),
                coordinates: vertex.point().coords().to_vec(),
                attributes: None,
            })
            .collect();
        vertices.sort_by_key(|record| record.id);

        let mut simplices = Vec::with_capacity(self.number_of_simplices());
        let mut adjacency =
            Vec::with_capacity(self.number_of_simplices().saturating_mul(simplex_arity));
        for (_, simplex) in self.simplices() {
            let simplex_id = simplex.uuid();
            if let Some(periodic) = &mut periodic {
                let offsets = simplex.periodic_vertex_offsets().map_or_else(
                    || vec![vec![0; D]; simplex.vertices().len()],
                    |offsets| offsets.iter().map(|offset| offset.to_vec()).collect(),
                );
                periodic.simplex_vertex_offsets.insert(simplex_id, offsets);
            }
            let vertex_ids = simplex
                .vertices()
                .iter()
                .copied()
                .map(|vertex_key| {
                    self.vertex(vertex_key).map(Vertex::uuid).ok_or(
                        VisualizationExportError::MissingVertex {
                            simplex_id,
                            vertex_key,
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            simplices.push(SimplexRecord {
                id: simplex_id,
                vertex_ids,
                attributes: None,
            });
            push_adjacency_records(
                |neighbor_key| self.simplex_uuid_from_key(neighbor_key),
                simplex_arity,
                simplex_id,
                simplex.neighbor_slots().map(NeighborBuffer::as_slice),
                &mut adjacency,
            )?;
        }
        simplices.sort_by_key(|record| record.id);
        adjacency.sort_by_key(|record| (record.simplex_id, record.facet_index));

        Ok(VisualizationData {
            metadata: VisualizationMetadata {
                schema: VISUALIZATION_SCHEMA.to_owned(),
                schema_version: VISUALIZATION_SCHEMA_VERSION,
                producer: env!("CARGO_PKG_NAME").to_owned(),
                dimension: D,
                vertex_count: vertices.len(),
                simplex_count: simplices.len(),
                topology_kind: VisualizationTopologyKind::from(self.topology_kind()),
                topology_guarantee: VisualizationTopologyGuarantee::from(self.topology_guarantee()),
                attributes: None,
            },
            vertices,
            simplices,
            adjacency,
            periodic,
        })
    }
}

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D> {
    /// Exports the same UUID-based records as [`Triangulation::to_visualization_data`].
    ///
    /// Export reads Levels 1–4 only and performs no Delaunay certification or repair.
    ///
    /// # Errors
    ///
    /// Returns [`VisualizationExportError`] for unresolved topology relationships.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};
    /// # fn main() -> DelaunayResult<()> {
    /// let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// assert_eq!(dt.to_visualization_data()?.vertices.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_visualization_data(&self) -> Result<VisualizationData<D>, VisualizationExportError> {
        self.as_triangulation().to_visualization_data()
    }
}

/// Canonicalizes detached record order for deterministic validated exports.
fn canonicalize_record_order<
    const D: usize,
    VertexAttributes,
    SimplexAttributes,
    AdjacencyAttributes,
    GlobalAttributes,
>(
    data: &mut VisualizationData<
        D,
        VertexAttributes,
        SimplexAttributes,
        AdjacencyAttributes,
        GlobalAttributes,
    >,
) {
    data.vertices.sort_by_key(|record| record.id);
    data.simplices.sort_by_key(|record| record.id);
    data.adjacency
        .sort_by_key(|record| (record.simplex_id, record.facet_index));
}

/// Rejects unrepresentable dimensions even when every entity table is empty.
fn visualization_arity<const D: usize>() -> Result<usize, VisualizationDataValidationError> {
    D.checked_add(1)
        .ok_or(VisualizationDataValidationError::DimensionOutOfRange { dimension: D })
}

/// Checks the periodic table before any vertex slot indexes its lifted offset.
fn validate_periodic_metadata<const D: usize, Attributes, SimplexAttributes>(
    metadata: &VisualizationMetadata<Attributes>,
    periodic: Option<&PeriodicVisualizationData>,
    simplices: &[SimplexRecord<SimplexAttributes>],
    simplex_arity: usize,
) -> Result<(), VisualizationDataValidationError> {
    let toroidal = metadata.topology_kind == VisualizationTopologyKind::Toroidal;
    if toroidal && (periodic.is_none() || metadata.schema_version < 2) {
        return Err(VisualizationDataValidationError::MissingPeriodicData {});
    }
    let Some(periodic) = periodic else {
        return Ok(());
    };
    if !toroidal || metadata.schema_version < 2 {
        return Err(VisualizationDataValidationError::UnexpectedPeriodicData {});
    }
    if periodic.periods.len() != D {
        return Err(VisualizationDataValidationError::InvalidPeriodCount {
            expected: D,
            actual: periodic.periods.len(),
        });
    }
    for (axis, period) in periodic.periods.iter().enumerate() {
        if !period.is_finite() || *period <= 0.0 {
            return Err(VisualizationDataValidationError::InvalidPeriod { axis });
        }
    }
    let simplex_ids: HashSet<_> = simplices.iter().map(|simplex| simplex.id).collect();
    for simplex_id in periodic.simplex_vertex_offsets.keys() {
        if !simplex_ids.contains(simplex_id) {
            return Err(VisualizationDataValidationError::UnknownPeriodicSimplex {
                simplex_id: *simplex_id,
            });
        }
    }
    for simplex in simplices {
        let offsets = periodic.simplex_vertex_offsets.get(&simplex.id).ok_or(
            VisualizationDataValidationError::MissingPeriodicOffsets {
                simplex_id: simplex.id,
            },
        )?;
        if offsets.len() != simplex_arity {
            return Err(
                VisualizationDataValidationError::InvalidPeriodicOffsetCount {
                    simplex_id: simplex.id,
                    expected: simplex_arity,
                    actual: offsets.len(),
                },
            );
        }
        for (vertex_index, offset) in offsets.iter().enumerate() {
            if offset.len() != D {
                return Err(
                    VisualizationDataValidationError::InvalidPeriodicOffsetDimension {
                        simplex_id: simplex.id,
                        vertex_index,
                        expected: D,
                        actual: offset.len(),
                    },
                );
            }
        }
    }
    Ok(())
}

/// Validates whole-payload schema metadata before walking entity records.
fn validate_metadata<const D: usize, Attributes>(
    metadata: &VisualizationMetadata<Attributes>,
    vertex_count: usize,
    simplex_count: usize,
) -> Result<(), VisualizationDataValidationError> {
    if metadata.schema != VISUALIZATION_SCHEMA {
        return Err(VisualizationDataValidationError::InvalidSchema {
            expected: VISUALIZATION_SCHEMA,
            actual: metadata.schema.clone(),
        });
    }
    if !matches!(metadata.schema_version, 1 | VISUALIZATION_SCHEMA_VERSION) {
        return Err(VisualizationDataValidationError::InvalidSchemaVersion {
            expected: VISUALIZATION_SCHEMA_VERSION,
            actual: metadata.schema_version,
        });
    }
    if metadata.dimension != D {
        return Err(VisualizationDataValidationError::DimensionMismatch {
            expected: D,
            actual: metadata.dimension,
        });
    }
    if metadata.vertex_count != vertex_count {
        return Err(VisualizationDataValidationError::VertexCountMismatch {
            expected: metadata.vertex_count,
            actual: vertex_count,
        });
    }
    if metadata.simplex_count != simplex_count {
        return Err(VisualizationDataValidationError::SimplexCountMismatch {
            expected: metadata.simplex_count,
            actual: simplex_count,
        });
    }
    if !metadata.topology_kind.is_supported() {
        return Err(VisualizationDataValidationError::InvalidTopologyKind {
            actual: metadata.topology_kind.clone(),
        });
    }
    if !metadata.topology_guarantee.is_supported() {
        return Err(VisualizationDataValidationError::InvalidTopologyGuarantee {
            actual: metadata.topology_guarantee.clone(),
        });
    }

    Ok(())
}

/// Validates that adjacency records reference known simplices, share their
/// claimed source facet, cover every facet once, and reciprocate neighbors.
fn validate_adjacency<const D: usize, SimplexAttributes, AdjacencyAttributes>(
    adjacency: &[AdjacencyRecord<AdjacencyAttributes>],
    simplices: &[SimplexRecord<SimplexAttributes>],
    periodic: Option<&PeriodicVisualizationData>,
) -> Result<(), VisualizationDataValidationError> {
    let max_exclusive = visualization_arity::<D>()?;
    let simplex_by_id: HashMap<_, _> = simplices
        .iter()
        .map(|simplex| (simplex.id, simplex))
        .collect();
    let mut adjacency_slots = HashSet::with_capacity(adjacency.len());
    let mut neighbor_edge_counts = HashMap::with_capacity(adjacency.len());

    for record in adjacency {
        let Some(source_simplex) = simplex_by_id.get(&record.simplex_id) else {
            return Err(VisualizationDataValidationError::MissingAdjacencySimplex {
                simplex_id: record.simplex_id,
            });
        };
        if record.facet_index >= max_exclusive {
            return Err(
                VisualizationDataValidationError::InvalidAdjacencyFacetIndex {
                    simplex_id: record.simplex_id,
                    facet_index: record.facet_index,
                    max_exclusive,
                },
            );
        }
        if let Some(neighbor_simplex_id) = record.neighbor_simplex_id {
            let Some(neighbor_simplex) = simplex_by_id.get(&neighbor_simplex_id) else {
                return Err(VisualizationDataValidationError::MissingAdjacencyNeighbor {
                    simplex_id: record.simplex_id,
                    facet_index: record.facet_index,
                    neighbor_simplex_id,
                });
            };
            if let Some(missing_vertex_id) =
                missing_source_facet_vertex(source_simplex, neighbor_simplex, record.facet_index)
            {
                return Err(
                    VisualizationDataValidationError::InvalidAdjacencyFacetSharing {
                        simplex_id: record.simplex_id,
                        facet_index: record.facet_index,
                        neighbor_simplex_id,
                        missing_vertex_id,
                    },
                );
            }
            *neighbor_edge_counts
                .entry((record.simplex_id, neighbor_simplex_id))
                .or_insert(0) += 1;
        }
        if !adjacency_slots.insert((record.simplex_id, record.facet_index)) {
            return Err(VisualizationDataValidationError::DuplicateAdjacency {
                simplex_id: record.simplex_id,
                facet_index: record.facet_index,
            });
        }
    }

    for simplex in simplices {
        for facet_index in 0..max_exclusive {
            if !adjacency_slots.contains(&(simplex.id, facet_index)) {
                return Err(VisualizationDataValidationError::MissingAdjacency {
                    simplex_id: simplex.id,
                    facet_index,
                });
            }
        }
    }

    for record in adjacency {
        if let Some(neighbor_simplex_id) = record.neighbor_simplex_id
            && !has_reciprocal_adjacency(
                &neighbor_edge_counts,
                record.simplex_id,
                neighbor_simplex_id,
            )
        {
            return Err(VisualizationDataValidationError::AsymmetricAdjacency {
                simplex_id: record.simplex_id,
                facet_index: record.facet_index,
                neighbor_simplex_id,
            });
        }
    }

    if let Some(periodic) = periodic {
        validate_periodic_adjacency(adjacency, &simplex_by_id, periodic)?;
    }

    Ok(())
}

/// Matches periodic facets by lifted vertex identity, allowing chart translation.
fn validate_periodic_adjacency<SimplexAttributes, AdjacencyAttributes>(
    adjacency: &[AdjacencyRecord<AdjacencyAttributes>],
    simplex_by_id: &HashMap<Uuid, &SimplexRecord<SimplexAttributes>>,
    periodic: &PeriodicVisualizationData,
) -> Result<(), VisualizationDataValidationError> {
    let signatures: HashMap<_, _> = adjacency
        .iter()
        .map(|record| {
            let simplex = simplex_by_id[&record.simplex_id];
            let offsets = &periodic.simplex_vertex_offsets[&record.simplex_id];
            (
                (record.simplex_id, record.facet_index),
                periodic_facet_signature(simplex, offsets, record.facet_index),
            )
        })
        .collect();
    let mut facets_by_simplex = HashMap::<_, Vec<_>>::new();
    for record in adjacency {
        facets_by_simplex
            .entry(record.simplex_id)
            .or_default()
            .push(record);
    }
    for record in adjacency {
        let Some(neighbor_simplex_id) = record.neighbor_simplex_id else {
            continue;
        };
        let source_slot = (record.simplex_id, record.facet_index);
        let matches = facets_by_simplex[&neighbor_simplex_id]
            .iter()
            .filter(|candidate| {
                let candidate_slot = (candidate.simplex_id, candidate.facet_index);
                candidate_slot != source_slot
                    && candidate.neighbor_simplex_id == Some(record.simplex_id)
                    && signatures[&candidate_slot] == signatures[&source_slot]
            })
            .count();
        if matches != 1 {
            return Err(
                VisualizationDataValidationError::InvalidPeriodicFacetSharing {
                    simplex_id: record.simplex_id,
                    facet_index: record.facet_index,
                    neighbor_simplex_id,
                },
            );
        }
    }
    Ok(())
}

/// Removes a facet's common lattice translation without overflowing i8 offsets.
fn periodic_facet_signature<Attributes>(
    simplex: &SimplexRecord<Attributes>,
    offsets: &[Vec<i8>],
    facet_index: usize,
) -> Vec<(Uuid, Vec<i16>)> {
    let mut vertices: Vec<_> = simplex
        .vertex_ids
        .iter()
        .zip(offsets)
        .enumerate()
        .filter(|(index, _)| *index != facet_index)
        .map(|(_, (id, offset))| {
            (
                *id,
                offset
                    .iter()
                    .map(|value| i16::from(*value))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    vertices.sort_unstable();
    if let Some((_, anchor)) = vertices.first() {
        let anchor = anchor.clone();
        for (_, offset) in &mut vertices {
            for (value, origin) in offset.iter_mut().zip(&anchor) {
                *value -= origin;
            }
        }
    }
    vertices
}

/// Enforces the exported adjacency reciprocity contract.
///
/// Ordinary neighbors reciprocate when the neighbor simplex has any record
/// pointing back to the source simplex. For self-neighbors, the edge count must
/// exceed the current record because the record itself cannot satisfy
/// reciprocity.
fn has_reciprocal_adjacency(
    neighbor_edge_counts: &HashMap<(Uuid, Uuid), usize>,
    simplex_id: Uuid,
    neighbor_simplex_id: Uuid,
) -> bool {
    let reciprocal_count = neighbor_edge_counts
        .get(&(neighbor_simplex_id, simplex_id))
        .copied()
        .unwrap_or(0);

    if neighbor_simplex_id == simplex_id {
        reciprocal_count > 1
    } else {
        reciprocal_count > 0
    }
}

/// Finds the first source-facet vertex absent from the candidate neighbor simplex.
fn missing_source_facet_vertex<SimplexAttributes>(
    source_simplex: &SimplexRecord<SimplexAttributes>,
    neighbor_simplex: &SimplexRecord<SimplexAttributes>,
    facet_index: usize,
) -> Option<Uuid> {
    source_simplex
        .vertex_ids
        .iter()
        .enumerate()
        .filter(|(vertex_index, _)| *vertex_index != facet_index)
        .map(|(_, vertex_id)| *vertex_id)
        .find(|vertex_id| !neighbor_simplex.vertex_ids.contains(vertex_id))
}

/// Resolves facet-neighbor slots through an owner-level UUID lookup.
fn push_adjacency_records(
    simplex_uuid_from_key: impl Fn(SimplexKey) -> Option<Uuid>,
    expected_neighbor_count: usize,
    simplex_id: Uuid,
    neighbor_slots: Option<&[NeighborSlot]>,
    adjacency: &mut Vec<AdjacencyRecord>,
) -> Result<(), VisualizationExportError> {
    let slots =
        neighbor_slots.ok_or(VisualizationExportError::UnassignedNeighborBuffer { simplex_id })?;
    if slots.len() != expected_neighbor_count {
        return Err(VisualizationExportError::InvalidNeighborCount {
            simplex_id,
            expected: expected_neighbor_count,
            actual: slots.len(),
        });
    }

    for (facet_index, slot) in slots.iter().copied().enumerate() {
        let record = match slot {
            NeighborSlot::Boundary => AdjacencyRecord {
                simplex_id,
                facet_index,
                neighbor_simplex_id: None,
                attributes: None,
            },
            NeighborSlot::Neighbor(neighbor_key) => simplex_uuid_from_key(neighbor_key)
                .map(|neighbor_simplex_id| AdjacencyRecord {
                    simplex_id,
                    facet_index,
                    neighbor_simplex_id: Some(neighbor_simplex_id),
                    attributes: None,
                })
                .ok_or(VisualizationExportError::MissingNeighbor {
                    simplex_id,
                    facet_index,
                    neighbor_key,
                })?,
            NeighborSlot::Unassigned => {
                return Err(VisualizationExportError::UnassignedNeighborSlot {
                    simplex_id,
                    facet_index,
                });
            }
        };
        adjacency.push(record);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::Tds;
    use crate::geometry::kernel::RobustKernel;
    use crate::topology::traits::topological_space::ToroidalConstructionMode;
    use crate::triangulation::builder::TriangulationBuilder;
    use slotmap::KeyData;

    #[test]
    fn periodic_export_preserves_implicit_zero_offset_charts() {
        let (mut tds, _) = Tds::periodic_grid_t2_for_test(false);
        let expected_offsets: BTreeMap<_, _> = tds
            .simplices()
            .map(|(_, simplex)| {
                (
                    simplex.uuid(),
                    simplex
                        .periodic_vertex_offsets()
                        .unwrap()
                        .iter()
                        .map(|offset| offset.to_vec())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let simplex_keys: Vec<_> = tds.simplex_keys().collect();
        let mut implicit_charts = 0;
        for simplex_key in simplex_keys {
            let simplex = tds.simplex_mut(simplex_key).unwrap();
            if simplex
                .periodic_vertex_offsets()
                .unwrap()
                .iter()
                .all(|offset| *offset == [0; 2])
            {
                simplex.periodic_vertex_offsets = None;
                implicit_charts += 1;
            }
        }
        assert_eq!(implicit_charts, 4);
        let topology =
            GlobalTopology::try_toroidal([1.0; 2], ToroidalConstructionMode::PeriodicImagePoint)
                .unwrap();
        let tri = TriangulationBuilder::new(tds, RobustKernel::new())
            .global_topology(topology)
            .build()
            .unwrap();
        tri.validate_realization().unwrap();
        let export = tri.to_visualization_data().unwrap();
        export.validate().unwrap();
        assert_eq!(
            export.periodic.unwrap().simplex_vertex_offsets,
            expected_offsets
        );
    }

    #[test]
    fn empty_visualization_rejects_dimension_overflow_before_validation() {
        let data = VisualizationData::<{ usize::MAX }> {
            metadata: VisualizationMetadata {
                schema: VISUALIZATION_SCHEMA.to_owned(),
                schema_version: VISUALIZATION_SCHEMA_VERSION,
                producer: "test".to_owned(),
                dimension: usize::MAX,
                vertex_count: 0,
                simplex_count: 0,
                topology_kind: VisualizationTopologyKind::Euclidean,
                topology_guarantee: VisualizationTopologyGuarantee::PLManifold,
                attributes: None,
            },
            vertices: Vec::new(),
            simplices: Vec::new(),
            adjacency: Vec::new(),
            periodic: None,
        };
        let expected = VisualizationDataValidationError::DimensionOutOfRange {
            dimension: usize::MAX,
        };
        assert_eq!(data.validate(), Err(expected.clone()));
        assert_eq!(data.into_validated(), Err(expected.clone()));
        assert_eq!(
            validate_adjacency::<{ usize::MAX }, (), ()>(&[], &[], None),
            Err(expected)
        );
    }

    #[test]
    fn push_adjacency_records_rejects_wrong_neighbor_arity() {
        let simplex_id = Uuid::from_u128(0x3000_0000_0000_0000_0000_0000_0000_0001);
        let mut adjacency = Vec::new();

        let error = push_adjacency_records(
            |_| None,
            3,
            simplex_id,
            Some(&[NeighborSlot::Boundary]),
            &mut adjacency,
        )
        .expect_err("neighbor buffers must have one slot per simplex facet");

        assert_eq!(
            error,
            VisualizationExportError::InvalidNeighborCount {
                simplex_id,
                expected: 3,
                actual: 1,
            }
        );
        assert_eq!(adjacency.as_slice(), []);
    }

    #[test]
    fn push_adjacency_records_rejects_unassigned_neighbor_slot() {
        let simplex_id = Uuid::from_u128(0x3000_0000_0000_0000_0000_0000_0000_0002);
        let mut adjacency = Vec::new();

        let error = push_adjacency_records(
            |_| None,
            3,
            simplex_id,
            Some(&[
                NeighborSlot::Unassigned,
                NeighborSlot::Boundary,
                NeighborSlot::Boundary,
            ]),
            &mut adjacency,
        )
        .expect_err("export must reject explicit unassigned neighbor slots");

        assert_eq!(
            error,
            VisualizationExportError::UnassignedNeighborSlot {
                simplex_id,
                facet_index: 0,
            }
        );
        assert_eq!(adjacency.as_slice(), []);
    }

    #[test]
    fn push_adjacency_records_rejects_dangling_neighbor_key() {
        let simplex_id = Uuid::from_u128(0x3000_0000_0000_0000_0000_0000_0000_0003);
        let neighbor_key = SimplexKey::from(KeyData::from_ffi(1));
        let mut adjacency = Vec::new();

        let error = push_adjacency_records(
            |_| None,
            3,
            simplex_id,
            Some(&[
                NeighborSlot::Neighbor(neighbor_key),
                NeighborSlot::Boundary,
                NeighborSlot::Boundary,
            ]),
            &mut adjacency,
        )
        .expect_err("export must reject neighbor keys absent from the TDS");

        assert_eq!(
            error,
            VisualizationExportError::MissingNeighbor {
                simplex_id,
                facet_index: 0,
                neighbor_key,
            }
        );
        assert_eq!(adjacency.as_slice(), []);
    }
}
