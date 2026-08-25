//! Versioned, integrity-checked persistence for Delaunay triangulations.
//!
//! Schema-v2 checkpoints carry a
//! [`DelaunayCheckpointManifest`](crate::checkpoint::DelaunayCheckpointManifest) beside the
//! canonical TDS snapshot and proof context. The manifest is untrusted evidence:
//! loading verifies it, rebuilds canonical storage, recomputes its topology
//! metrics, and then re-establishes validation Levels 1–5. No manifest field is
//! used to choose a reconstruction path or skip validation.

#![forbid(unsafe_code)]

use crate::DelaunayTriangulationConstructionError;
use crate::builder::DelaunayTriangulationBuilder;
use crate::core::simplex::SimplexValidationError;
use crate::core::tds::{
    RawTdsSnapshot, SimplexKey, Tds, TdsError, TdsMutationError, TdsSnapshot, TdsSnapshotError,
    ValidatedTdsSerialization, VertexKey,
};
use crate::core::traits::data_type::{DataDeserialize, DataSerialize, DataType};
use crate::core::vertex::Vertex;
use crate::delaunay_model::DelaunayTriangulation;
use crate::geometry::kernel::{ExactPredicates, Kernel, RobustKernel};
use crate::topology::characteristics::euler::{count_simplices, euler_characteristic};
use crate::topology::traits::topological_space::{
    GlobalTopology, TopologyError, ToroidalConstructionMode, ToroidalDomainError,
};
use crate::triangulation::builder::TriangulationBuilderError;
use crate::triangulation::validation::{
    TopologyCertificationEvidence, TopologyConstructionProvenance, TopologyGuarantee,
    ValidationConfigurationError, ValidationPolicy,
};
use crate::validation::{
    DelaunayTdsRestorationError, DelaunayTdsRestorationReason, DelaunayTriangulationValidationError,
};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, SeqAccess, Visitor},
    ser::{
        self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
        SerializeTupleStruct, SerializeTupleVariant,
    },
};
use sha2::{Digest, Sha256};
use thiserror::Error;
use uuid::Uuid;

use ciborium::value::Value;

use std::{error::Error as StdError, fmt, marker::PhantomData, mem::size_of, str};

// =============================================================================
// CHECKPOINT VERSIONS AND PUBLIC MANIFEST TYPES
// =============================================================================

/// Current owner-checkpoint schema version.
///
/// Ordinary loading rejects schema v1 because it has no scientific integrity
/// manifest. Use [`DelaunayCheckpointV1`] explicitly to validate and migrate a
/// legacy owner, then serialize the returned owner as schema v2.
pub const DELAUNAY_CHECKPOINT_SCHEMA_VERSION: u32 = 2;

/// Current scientific-manifest shape version.
pub const DELAUNAY_CHECKPOINT_MANIFEST_VERSION: u32 = 1;

/// Current canonical checkpoint representation and digest version.
pub const DELAUNAY_CHECKPOINT_DIGEST_VERSION: u32 = 1;

/// Digest algorithm used by schema-v2 scientific manifests.
pub const DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM: &str = "sha256";

const SHA256_HEX_LENGTH: usize = 64;
const CANONICAL_CHECKPOINT_DOMAIN: &[u8] = b"delaunay.checkpoint.canonical\0";

/// Open digest-algorithm identifier used by checkpoint manifests.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DelaunayCheckpointDigestAlgorithm {
    /// SHA-256 over the crate-defined canonical representation.
    Sha256,
    /// An identifier unknown to this crate version.
    ///
    /// Construct open identifiers through [`Self::from_identifier`] so the
    /// current durable name cannot be misclassified as unknown.
    ///
    /// ```compile_fail
    /// use delaunay::prelude::checkpoint::DelaunayCheckpointDigestAlgorithm;
    ///
    /// let _ = DelaunayCheckpointDigestAlgorithm::Unknown("sha256".to_owned());
    /// ```
    #[non_exhaustive]
    Unknown(String),
}

impl DelaunayCheckpointDigestAlgorithm {
    /// Canonicalizes one durable algorithm identifier.
    ///
    /// The current SHA-256 identifier always produces [`Self::Sha256`]; every
    /// other identifier is retained as an open, forward-compatible value.
    #[must_use]
    pub fn from_identifier(identifier: impl Into<String>) -> Self {
        let identifier = identifier.into();
        if identifier == DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM {
            Self::Sha256
        } else {
            Self::Unknown(identifier)
        }
    }

    /// Returns the exact durable wire identifier.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::checkpoint::{
    ///     DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM, DelaunayCheckpointDigestAlgorithm,
    /// };
    ///
    /// assert_eq!(
    ///     DelaunayCheckpointDigestAlgorithm::Sha256.as_str(),
    ///     DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM,
    /// );
    /// assert_eq!(
    ///     DelaunayCheckpointDigestAlgorithm::from_identifier("future-digest").as_str(),
    ///     "future-digest",
    /// );
    /// assert_eq!(
    ///     DelaunayCheckpointDigestAlgorithm::from_identifier(
    ///         DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM,
    ///     ),
    ///     DelaunayCheckpointDigestAlgorithm::Sha256,
    /// );
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Sha256 => DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM,
            Self::Unknown(identifier) => identifier,
        }
    }
}

impl fmt::Display for DelaunayCheckpointDigestAlgorithm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for DelaunayCheckpointDigestAlgorithm {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DelaunayCheckpointDigestAlgorithm {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let identifier = String::deserialize(deserializer)?;
        Ok(Self::from_identifier(identifier))
    }
}

/// Versioned cryptographic digest of a canonical Delaunay checkpoint.
///
/// `value` is a 64-character lowercase hexadecimal SHA-256 digest. The digest
/// version identifies the crate-defined canonical byte representation, not the
/// JSON, CBOR, or other outer serde codec.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct DelaunayCheckpointDigest {
    /// Canonical-representation and digest version.
    pub version: u32,
    /// Digest algorithm identifier.
    pub algorithm: DelaunayCheckpointDigestAlgorithm,
    /// Lowercase hexadecimal digest value.
    pub value: String,
}

/// Scientific and integrity evidence stored with one Delaunay checkpoint.
///
/// The f-vector records face counts for the current subdivision and can change
/// under a valid bistellar move; it does not uniquely identify connectivity.
/// Its alternating sum, the Euler characteristic, is a PL invariant preserved
/// by such moves. Both are recomputed through the Level-3 topology
/// implementation when a checkpoint is loaded.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct DelaunayCheckpointManifest {
    /// Scientific-manifest shape version.
    pub manifest_version: u32,
    /// Compile-time simplicial dimension of the serialized owner.
    pub dimension: u32,
    /// Complete f-vector `(f_0, ..., f_D)` for the exact subdivision.
    pub f_vector: Vec<u64>,
    /// Alternating sum of the complete f-vector.
    pub euler_characteristic: i64,
    /// Digest of the canonical checkpoint representation.
    pub digest: DelaunayCheckpointDigest,
}

// =============================================================================
// BOUNDED OWNER WIRE TYPES
// =============================================================================

/// Owner-level persistence DTO that keeps proof context beside canonical TDS storage.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelaunayTriangulationWire<const D: usize> {
    schema_version: u32,
    manifest: Option<DecodedDelaunayCheckpointManifest<D>>,
    tds: TdsBytes,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
}

/// Serialize-only wire keeps schema-v2's manifest mandatory in emitted data.
#[derive(Serialize)]
struct DelaunayTriangulationSerializeWire<const D: usize> {
    schema_version: u32,
    manifest: DelaunayCheckpointManifestWire<D>,
    tds: TdsBytes,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
}

/// Exact-length sequence used at untrusted fixed-arity checkpoint boundaries.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
struct ExactSlots<T, const N: usize>(Vec<T>);

impl<'de, T, const N: usize> Deserialize<'de> for ExactSlots<T, N>
where
    T: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct ExactSlotsVisitor<T, const N: usize>(PhantomData<T>);

        impl<'de, T, const N: usize> Visitor<'de> for ExactSlotsVisitor<T, N>
        where
            T: Deserialize<'de>,
        {
            type Value = ExactSlots<T, N>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a sequence with exactly {N} entries")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(N);
                while values.len() < N {
                    let Some(value) = sequence.next_element()? else {
                        return Err(de::Error::invalid_length(values.len(), &self));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(N + 1, &self));
                }
                Ok(ExactSlots(values))
            }
        }

        deserializer.deserialize_seq(ExactSlotsVisitor(PhantomData))
    }
}

/// Exact `D + 1` sequence used for manifest face counts.
#[derive(Clone, Debug, Serialize)]
#[serde(transparent)]
struct DPlusOneSlots<T, const D: usize>(Vec<T>);

impl<'de, T, const D: usize> Deserialize<'de> for DPlusOneSlots<T, D>
where
    T: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct DPlusOneVisitor<T, const D: usize>(PhantomData<T>);

        impl<'de, T, const D: usize> Visitor<'de> for DPlusOneVisitor<T, D>
        where
            T: Deserialize<'de>,
        {
            type Value = DPlusOneSlots<T, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "a sequence with exactly {} entries", D + 1)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let expected = D + 1;
                let mut values = Vec::with_capacity(expected);
                while values.len() < expected {
                    let Some(value) = sequence.next_element()? else {
                        return Err(de::Error::invalid_length(values.len(), &self));
                    };
                    values.push(value);
                }
                if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                    return Err(de::Error::invalid_length(expected + 1, &self));
                }
                Ok(DPlusOneSlots(values))
            }
        }

        deserializer.deserialize_seq(DPlusOneVisitor(PhantomData))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DelaunayCheckpointManifestWire<const D: usize> {
    manifest_version: u32,
    dimension: u32,
    f_vector: DPlusOneSlots<u64, D>,
    euler_characteristic: i64,
    digest: DelaunayCheckpointDigest,
}

impl<const D: usize> From<DelaunayCheckpointManifest> for DelaunayCheckpointManifestWire<D> {
    fn from(value: DelaunayCheckpointManifest) -> Self {
        Self {
            manifest_version: value.manifest_version,
            dimension: value.dimension,
            f_vector: DPlusOneSlots(value.f_vector),
            euler_characteristic: value.euler_characteristic,
            digest: value.digest,
        }
    }
}

impl<const D: usize> From<DelaunayCheckpointManifestWire<D>> for DelaunayCheckpointManifest {
    fn from(value: DelaunayCheckpointManifestWire<D>) -> Self {
        Self {
            manifest_version: value.manifest_version,
            dimension: value.dimension,
            f_vector: value.f_vector.0,
            euler_characteristic: value.euler_characteristic,
            digest: value.digest,
        }
    }
}

/// Manifest decoded through the dimension-bounded wire shape for borrowed inspection.
#[derive(Debug)]
struct DecodedDelaunayCheckpointManifest<const D: usize>(DelaunayCheckpointManifest);

impl<const D: usize> From<DelaunayCheckpointManifest> for DecodedDelaunayCheckpointManifest<D> {
    fn from(value: DelaunayCheckpointManifest) -> Self {
        Self(value)
    }
}

impl<'de, const D: usize> Deserialize<'de> for DecodedDelaunayCheckpointManifest<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let wire = DelaunayCheckpointManifestWire::<D>::deserialize(deserializer)?;
        Ok(Self(wire.into()))
    }
}

/// Embedded CBOR TDS image. The outer codec only sees bytes.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
struct TdsBytes(Vec<u8>);

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum TopologyGuaranteeWire {
    Pseudomanifold,
    PlManifold,
}

impl From<TopologyGuarantee> for TopologyGuaranteeWire {
    fn from(value: TopologyGuarantee) -> Self {
        match value {
            TopologyGuarantee::Pseudomanifold => Self::Pseudomanifold,
            TopologyGuarantee::PLManifold => Self::PlManifold,
        }
    }
}

impl From<TopologyGuaranteeWire> for TopologyGuarantee {
    fn from(value: TopologyGuaranteeWire) -> Self {
        match value {
            TopologyGuaranteeWire::Pseudomanifold => Self::Pseudomanifold,
            TopologyGuaranteeWire::PlManifold => Self::PLManifold,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ValidationPolicyWire {
    Never,
    ExplicitOnly,
    OnSuspicion,
    Always,
    DebugOnly,
}

impl From<ValidationPolicy> for ValidationPolicyWire {
    fn from(value: ValidationPolicy) -> Self {
        match value {
            ValidationPolicy::Never => Self::Never,
            ValidationPolicy::ExplicitOnly => Self::ExplicitOnly,
            ValidationPolicy::OnSuspicion => Self::OnSuspicion,
            ValidationPolicy::Always => Self::Always,
            ValidationPolicy::DebugOnly => Self::DebugOnly,
        }
    }
}

impl From<ValidationPolicyWire> for ValidationPolicy {
    fn from(value: ValidationPolicyWire) -> Self {
        match value {
            ValidationPolicyWire::Never => Self::Never,
            ValidationPolicyWire::ExplicitOnly => Self::ExplicitOnly,
            ValidationPolicyWire::OnSuspicion => Self::OnSuspicion,
            ValidationPolicyWire::Always => Self::Always,
            ValidationPolicyWire::DebugOnly => Self::DebugOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToroidalConstructionModeWire {
    PeriodicImagePoint,
    Explicit,
}

impl From<ToroidalConstructionMode> for ToroidalConstructionModeWire {
    fn from(value: ToroidalConstructionMode) -> Self {
        match value {
            ToroidalConstructionMode::PeriodicImagePoint => Self::PeriodicImagePoint,
            ToroidalConstructionMode::Explicit => Self::Explicit,
        }
    }
}

impl From<ToroidalConstructionModeWire> for ToroidalConstructionMode {
    fn from(value: ToroidalConstructionModeWire) -> Self {
        match value {
            ToroidalConstructionModeWire::PeriodicImagePoint => Self::PeriodicImagePoint,
            ToroidalConstructionModeWire::Explicit => Self::Explicit,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum GlobalTopologyWire<const D: usize> {
    Euclidean {},
    Toroidal {
        periods: ExactSlots<u64, D>,
        mode: ToroidalConstructionModeWire,
    },
    Spherical {},
    Hyperbolic {},
}

impl<const D: usize> From<GlobalTopology<D>> for GlobalTopologyWire<D> {
    fn from(value: GlobalTopology<D>) -> Self {
        match value {
            GlobalTopology::Euclidean => Self::Euclidean {},
            GlobalTopology::Toroidal { domain, mode } => Self::Toroidal {
                periods: ExactSlots(
                    domain
                        .periods()
                        .iter()
                        .map(|period| period.to_bits())
                        .collect(),
                ),
                mode: mode.into(),
            },
            GlobalTopology::Spherical => Self::Spherical {},
            GlobalTopology::Hyperbolic => Self::Hyperbolic {},
        }
    }
}

impl<const D: usize> GlobalTopologyWire<D> {
    /// Parses raw topology metadata through the domain constructor before proof restoration.
    fn try_into_global_topology(self) -> Result<GlobalTopology<D>, DelaunayCheckpointError> {
        match self {
            Self::Euclidean {} => Ok(GlobalTopology::Euclidean),
            Self::Toroidal { periods, mode } => {
                let actual = periods.0.len();
                let period_bits: [u64; D] = periods.0.try_into().map_err(|_| {
                    DelaunayCheckpointError::ToroidalDimensionMismatch {
                        expected: D,
                        actual,
                    }
                })?;
                let periods = period_bits.map(f64::from_bits);
                GlobalTopology::try_toroidal(periods, mode.into())
                    .map_err(|source| DelaunayCheckpointError::ToroidalDomain { source })
            }
            Self::Spherical {} => Ok(GlobalTopology::Spherical),
            Self::Hyperbolic {} => Ok(GlobalTopology::Hyperbolic),
        }
    }
}

/// Legacy schema-v1 topology metadata used only by the explicit migrator.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum LegacyGlobalTopologyWire<const D: usize> {
    Euclidean {},
    Toroidal {
        periods: ExactSlots<f64, D>,
        mode: ToroidalConstructionModeWire,
    },
    Spherical {},
    Hyperbolic {},
}

impl<const D: usize> LegacyGlobalTopologyWire<D> {
    /// Parses legacy floating-point periods through the current topology constructor.
    ///
    /// This preserves the schema-v1 migration contract: old wire data must
    /// satisfy current toroidal-domain invariants before any proof is restored.
    fn try_into_global_topology(self) -> Result<GlobalTopology<D>, DelaunayCheckpointError> {
        match self {
            Self::Euclidean {} => Ok(GlobalTopology::Euclidean),
            Self::Toroidal { periods, mode } => {
                let actual = periods.0.len();
                let periods = periods.0.try_into().map_err(|_| {
                    DelaunayCheckpointError::ToroidalDimensionMismatch {
                        expected: D,
                        actual,
                    }
                })?;
                GlobalTopology::try_toroidal(periods, mode.into())
                    .map_err(|source| DelaunayCheckpointError::ToroidalDomain { source })
            }
            Self::Spherical {} => Ok(GlobalTopology::Spherical),
            Self::Hyperbolic {} => Ok(GlobalTopology::Hyperbolic),
        }
    }
}

/// Exact owner wire emitted by schema v1 before scientific manifests existed.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DelaunayTriangulationV1Wire<T, const D: usize> {
    schema_version: u32,
    tds: T,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: LegacyGlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
}

// =============================================================================
// CHECKPOINT ERROR TYPES
// =============================================================================

/// Typed semantic failure while hydrating an embedded checkpoint TDS image.
///
/// These variants preserve durable UUID relationship evidence without exposing
/// the crate-private snapshot representation used by the low-level TDS codec.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayCheckpointTdsHydrationError {
    /// A simplex could not resolve one of its runtime vertex keys to a UUID.
    #[error("could not resolve vertex UUIDs for simplex {simplex_uuid}: {source}")]
    SimplexVertexUuidResolutionFailed {
        /// Simplex whose vertex UUIDs could not be resolved.
        simplex_uuid: Uuid,
        /// Typed simplex validation failure.
        #[source]
        source: Box<SimplexValidationError>,
    },
    /// A simplex has no assigned neighbor slots.
    #[error("no assigned neighbor slots found for simplex {simplex_uuid}")]
    MissingSimplexNeighborSlots {
        /// Simplex whose neighbor slots are absent.
        simplex_uuid: Uuid,
    },
    /// A runtime neighbor key could not be resolved to a simplex UUID.
    #[error(
        "neighbor key {neighbor_key:?} referenced by simplex {simplex_uuid} was not found in the checkpoint simplices"
    )]
    DanglingRuntimeNeighborKey {
        /// Simplex containing the dangling runtime neighbor key.
        simplex_uuid: Uuid,
        /// Runtime neighbor key that could not be resolved.
        neighbor_key: SimplexKey,
    },
    /// A vertex UUID appeared more than once in the snapshot records.
    #[error("duplicate vertex UUID {vertex_uuid} in the checkpoint TDS image")]
    DuplicateVertexUuid {
        /// Duplicate vertex UUID.
        vertex_uuid: Uuid,
    },
    /// A simplex record has no matching vertex-UUID relationship entry.
    #[error("no vertex UUIDs found for simplex {simplex_uuid}")]
    MissingSimplexVertexUuids {
        /// Simplex missing its vertex-UUID relationship.
        simplex_uuid: Uuid,
    },
    /// A simplex record has no matching neighbor-UUID relationship entry.
    #[error("no neighbor UUIDs found for simplex {simplex_uuid}")]
    MissingSimplexNeighborUuids {
        /// Simplex missing its neighbor-UUID relationship.
        simplex_uuid: Uuid,
    },
    /// A simplex vertex-UUID relationship has the wrong number of slots.
    #[error(
        "simplex {simplex_uuid} has {actual} vertex UUID slots in the checkpoint TDS image; expected {expected}"
    )]
    InvalidSimplexVertexUuidSlotCount {
        /// Simplex whose vertex relationship has the wrong arity.
        simplex_uuid: Uuid,
        /// Observed number of vertex UUID slots.
        actual: usize,
        /// Required number of vertex UUID slots.
        expected: usize,
    },
    /// A simplex references a vertex UUID absent from the image.
    #[error("vertex UUID {vertex_uuid} referenced by simplex {simplex_uuid} was not found")]
    DanglingSimplexVertexUuid {
        /// Simplex containing the dangling vertex reference.
        simplex_uuid: Uuid,
        /// Vertex UUID that could not be resolved.
        vertex_uuid: Uuid,
    },
    /// A simplex references a neighbor UUID absent from the image.
    #[error("neighbor UUID {neighbor_uuid} referenced by simplex {simplex_uuid} was not found")]
    DanglingSimplexNeighborUuid {
        /// Simplex containing the dangling neighbor reference.
        simplex_uuid: Uuid,
        /// Neighbor UUID that could not be resolved.
        neighbor_uuid: Uuid,
    },
    /// A simplex could not be reconstructed from its resolved relationships.
    #[error("invalid checkpoint simplex {simplex_uuid}: {source}")]
    InvalidSimplex {
        /// Simplex that could not be reconstructed.
        simplex_uuid: Uuid,
        /// Typed simplex validation failure.
        #[source]
        source: Box<SimplexValidationError>,
    },
    /// A simplex UUID appeared more than once in the snapshot records.
    #[error("duplicate simplex UUID {simplex_uuid} in the checkpoint TDS image")]
    DuplicateSimplexUuid {
        /// Duplicate simplex UUID.
        simplex_uuid: Uuid,
    },
    /// The vertex relationship map mentions an unknown simplex.
    #[error("vertex UUID mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexVertexMapping {
        /// Unknown simplex UUID.
        simplex_uuid: Uuid,
    },
    /// The neighbor relationship map mentions an unknown simplex.
    #[error("neighbor UUID mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexNeighborMapping {
        /// Unknown simplex UUID.
        simplex_uuid: Uuid,
    },
    /// The periodic-offset relationship map mentions an unknown simplex.
    #[error("periodic offset mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexOffsetMapping {
        /// Unknown simplex UUID.
        simplex_uuid: Uuid,
    },
    /// A serialized periodic offset has the wrong coordinate dimension.
    #[error(
        "periodic offset {offset_index} for simplex {simplex_uuid} has dimension {actual}; expected {expected}"
    )]
    PeriodicOffsetDimensionMismatch {
        /// Simplex containing the malformed offset.
        simplex_uuid: Uuid,
        /// Offset index within the simplex-local list.
        offset_index: usize,
        /// Required coordinate dimension.
        expected: usize,
        /// Observed coordinate dimension.
        actual: usize,
    },
    /// Rebuilding vertex incident-simplex pointers failed.
    #[error("failed to rebuild checkpoint TDS vertex incidence: {source}")]
    IncidentSimplexRebuildFailed {
        /// Typed TDS mutation failure.
        #[source]
        source: Box<TdsMutationError>,
    },
    /// Final Levels 1–2 validation failed after UUID relationships were resolved.
    #[error("hydrated checkpoint TDS failed validation: {source}")]
    ValidationFailed {
        /// Typed TDS validation failure.
        #[source]
        source: Box<TdsError>,
    },
}

impl From<TdsSnapshotError> for DelaunayCheckpointTdsHydrationError {
    fn from(source: TdsSnapshotError) -> Self {
        match source {
            TdsSnapshotError::SimplexVertexUuidResolutionFailed {
                simplex_uuid,
                source,
            } => Self::SimplexVertexUuidResolutionFailed {
                simplex_uuid,
                source: Box::new(source),
            },
            TdsSnapshotError::MissingSimplexNeighborSlots { simplex_uuid } => {
                Self::MissingSimplexNeighborSlots { simplex_uuid }
            }
            TdsSnapshotError::DanglingRuntimeNeighborKey {
                simplex_uuid,
                neighbor_key,
            } => Self::DanglingRuntimeNeighborKey {
                simplex_uuid,
                neighbor_key,
            },
            TdsSnapshotError::DuplicateVertexUuid { vertex_uuid } => {
                Self::DuplicateVertexUuid { vertex_uuid }
            }
            TdsSnapshotError::MissingSimplexVertexUuids { simplex_uuid } => {
                Self::MissingSimplexVertexUuids { simplex_uuid }
            }
            TdsSnapshotError::MissingSimplexNeighborUuids { simplex_uuid } => {
                Self::MissingSimplexNeighborUuids { simplex_uuid }
            }
            TdsSnapshotError::InvalidSimplexVertexUuidSlotCount {
                simplex_uuid,
                actual,
                expected,
            } => Self::InvalidSimplexVertexUuidSlotCount {
                simplex_uuid,
                actual,
                expected,
            },
            TdsSnapshotError::DanglingSimplexVertexUuid {
                simplex_uuid,
                vertex_uuid,
            } => Self::DanglingSimplexVertexUuid {
                simplex_uuid,
                vertex_uuid,
            },
            TdsSnapshotError::DanglingSimplexNeighborUuid {
                simplex_uuid,
                neighbor_uuid,
            } => Self::DanglingSimplexNeighborUuid {
                simplex_uuid,
                neighbor_uuid,
            },
            TdsSnapshotError::InvalidSimplex {
                simplex_uuid,
                source,
            } => Self::InvalidSimplex {
                simplex_uuid,
                source: Box::new(source),
            },
            TdsSnapshotError::DuplicateSimplexUuid { simplex_uuid } => {
                Self::DuplicateSimplexUuid { simplex_uuid }
            }
            TdsSnapshotError::UnknownSimplexVertexMapping { simplex_uuid } => {
                Self::UnknownSimplexVertexMapping { simplex_uuid }
            }
            TdsSnapshotError::UnknownSimplexNeighborMapping { simplex_uuid } => {
                Self::UnknownSimplexNeighborMapping { simplex_uuid }
            }
            TdsSnapshotError::UnknownSimplexOffsetMapping { simplex_uuid } => {
                Self::UnknownSimplexOffsetMapping { simplex_uuid }
            }
            TdsSnapshotError::PeriodicOffsetDimensionMismatch {
                simplex_uuid,
                offset_index,
                expected,
                actual,
            } => Self::PeriodicOffsetDimensionMismatch {
                simplex_uuid,
                offset_index,
                expected,
                actual,
            },
            TdsSnapshotError::IncidentSimplexRebuildFailed { source } => {
                Self::IncidentSimplexRebuildFailed {
                    source: Box::new(source),
                }
            }
            TdsSnapshotError::ValidationFailed { source } => Self::ValidationFailed {
                source: Box::new(source),
            },
        }
    }
}

/// Typed failures detected while building or verifying a Delaunay checkpoint.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayCheckpointError {
    /// Checkpoint owner schema is not supported by this crate version.
    #[error(
        "unsupported Delaunay triangulation serialization schema version {actual}; expected {expected}"
    )]
    UnsupportedSchemaVersion {
        /// Version found in the checkpoint.
        actual: u32,
        /// Version required by this crate.
        expected: u32,
    },
    /// Schema-v2 payload omitted its required manifest.
    #[error("Delaunay checkpoint schema v2 is missing its scientific manifest")]
    MissingManifest,
    /// Manifest shape version is unsupported.
    #[error("unsupported Delaunay checkpoint manifest version {actual}; expected {expected}")]
    UnsupportedManifestVersion {
        /// Version found in the manifest.
        actual: u32,
        /// Version required by this crate.
        expected: u32,
    },
    /// Manifest dimension disagrees with the compile-time dimension.
    #[error(
        "checkpoint manifest dimension {actual} does not match compile-time dimension {expected}"
    )]
    DimensionMismatch {
        /// Compile-time owner dimension.
        expected: u32,
        /// Dimension declared by the manifest.
        actual: u32,
    },
    /// Manifest f-vector has the wrong number of entries.
    #[error("checkpoint manifest f-vector has {actual} entries; expected {expected}")]
    FVectorLengthMismatch {
        /// Required number of entries (`D + 1`).
        expected: usize,
        /// Number of entries present in the manifest.
        actual: usize,
    },
    /// Manifest Euler evidence disagrees with its own f-vector.
    #[error(
        "checkpoint manifest is internally inconsistent: declared Euler characteristic {declared}, but its f-vector has alternating sum {derived}"
    )]
    ManifestEulerMismatch {
        /// Euler value stored in the manifest.
        declared: i64,
        /// Alternating sum derived from the manifest f-vector.
        derived: i64,
    },
    /// Manifest f-vector alternating sum cannot be represented as `i64`.
    #[error("checkpoint manifest f-vector alternating sum is outside the supported i64 range")]
    ManifestEulerOutOfRange,
    /// Canonical digest version is unsupported.
    #[error("unsupported Delaunay checkpoint digest version {actual}; expected {expected}")]
    UnsupportedDigestVersion {
        /// Digest version found in the manifest.
        actual: u32,
        /// Digest version required by this crate.
        expected: u32,
    },
    /// Digest algorithm identifier is unsupported.
    #[error("unsupported Delaunay checkpoint digest algorithm {actual}; expected {expected}")]
    UnsupportedDigestAlgorithm {
        /// Algorithm identifier found in the manifest.
        actual: DelaunayCheckpointDigestAlgorithm,
        /// Algorithm identifier required by this crate.
        expected: DelaunayCheckpointDigestAlgorithm,
    },
    /// Digest is not a lowercase hexadecimal SHA-256 value.
    #[error("malformed Delaunay checkpoint SHA-256 digest {actual:?}")]
    MalformedDigest {
        /// Malformed digest text from the manifest.
        actual: String,
    },
    /// Stored digest disagrees with the canonical checkpoint representation.
    #[error("Delaunay checkpoint digest mismatch: declared {declared}, computed {computed}")]
    DigestMismatch {
        /// Digest stored in the manifest.
        declared: String,
        /// Digest recomputed from canonical owner state.
        computed: String,
    },
    /// Stored f-vector disagrees with the recomputed Level-3 f-vector.
    #[error("Delaunay checkpoint f-vector mismatch: declared {declared:?}, computed {computed:?}")]
    FVectorMismatch {
        /// F-vector stored in the manifest.
        declared: Vec<u64>,
        /// F-vector recomputed through Level 3.
        computed: Vec<u64>,
    },
    /// Stored Euler characteristic disagrees with the recomputed value.
    #[error(
        "Delaunay checkpoint Euler characteristic mismatch: declared {declared}, computed {computed}"
    )]
    EulerCharacteristicMismatch {
        /// Euler characteristic stored in the manifest.
        declared: i64,
        /// Euler characteristic recomputed through Level 3.
        computed: i64,
    },
    /// Internal Level-3 evidence no longer describes the restored TDS owner.
    #[error("checkpoint topology evidence does not match the restored TDS generation")]
    StaleTopologyEvidence,
    /// A source TDS failed Levels 1–2 before canonical digest construction.
    #[error("cannot construct a canonical checkpoint from an invalid TDS: {source}")]
    TdsValidation {
        /// Levels 1–2 validation failure.
        #[source]
        source: Box<TdsError>,
    },
    /// Canonical payload conversion failed through the Serde data model.
    #[error("cannot canonicalize a checkpoint user payload: {message}")]
    PayloadSerialization {
        /// Serde serializer diagnostic supplied by the user payload.
        message: String,
    },
    /// A signed payload integer is outside digest-v1's canonical CBOR range.
    #[error("signed checkpoint payload integer {value} is outside the canonical CBOR range")]
    SignedPayloadIntegerOutOfRange {
        /// Signed integer rejected by the canonical payload encoder.
        value: i128,
    },
    /// An unsigned payload integer is outside digest-v1's canonical CBOR range.
    #[error("unsigned checkpoint payload integer {value} is outside the canonical CBOR range")]
    UnsignedPayloadIntegerOutOfRange {
        /// Unsigned integer rejected by the canonical payload encoder.
        value: u128,
    },
    /// A payload emitted an invalid Serde representation of a CBOR tag.
    #[error("cannot canonicalize a malformed CBOR tag in a checkpoint payload")]
    InvalidPayloadCborTag,
    /// A present payload uses Serde null/unit states that the canonical payload
    /// model cannot distinguish injectively.
    #[error(
        "cannot canonicalize a present checkpoint payload containing null or unit state; use a payload type with a distinct non-null representation"
    )]
    AmbiguousPayload,
    /// A payload map emitted the same canonical key more than once.
    #[error("cannot canonicalize a checkpoint payload map with duplicate canonical keys")]
    DuplicatePayloadMapKey,
    /// A sequence omitted its length, so digest-v1 could not stream it safely.
    #[error("cannot canonicalize a checkpoint payload sequence without a declared length")]
    UnboundedPayloadSequence,
    /// A Serde collection emitted a different number of values than declared.
    #[error("checkpoint payload collection declared {expected} entries but emitted {actual}")]
    PayloadCollectionLengthMismatch {
        /// Length declared when serialization began.
        expected: usize,
        /// Number of entries actually emitted.
        actual: usize,
    },
    /// A Serde map value was emitted without a preceding key, or vice versa.
    #[error("checkpoint payload map emitted an incomplete key/value pair")]
    IncompletePayloadMapEntry,
    /// Canonical payload model contains a value unknown to this digest version.
    #[error("cannot canonicalize a user payload value unknown to digest version 1")]
    UnsupportedPayloadValue,
    /// A length or count cannot be represented by the canonical u64 format.
    #[error("canonical checkpoint length or count {value} is outside the supported u64 range")]
    CanonicalLengthOutOfRange {
        /// Length that could not be represented.
        value: usize,
    },
    /// A runtime simplex vertex reference could not be resolved to a UUID.
    #[error("simplex {simplex_uuid} references missing vertex key {vertex_key:?}")]
    MissingVertexReference {
        /// Simplex containing the missing reference.
        simplex_uuid: Uuid,
        /// Storage-local key that did not resolve.
        vertex_key: VertexKey,
    },
    /// A runtime simplex has no assigned neighbor slots.
    #[error("simplex {simplex_uuid} has no assigned neighbor slots")]
    MissingNeighborSlots {
        /// Simplex lacking assigned neighbor slots.
        simplex_uuid: Uuid,
    },
    /// A runtime simplex neighbor reference could not be resolved to a UUID.
    #[error("simplex {simplex_uuid} references missing neighbor key {neighbor_key:?}")]
    MissingNeighborReference {
        /// Simplex containing the missing reference.
        simplex_uuid: Uuid,
        /// Storage-local neighbor key that did not resolve.
        neighbor_key: SimplexKey,
    },
    /// Compile-time dimension cannot be represented by the durable manifest.
    #[error("compile-time dimension {dimension} is outside the supported u32 range")]
    DimensionOutOfRange {
        /// Compile-time dimension that could not be represented.
        dimension: usize,
    },
    /// One f-vector entry cannot be represented by the durable manifest.
    #[error("f-vector entry {value} is outside the supported u64 range")]
    FVectorEntryOutOfRange {
        /// Simplex count that could not be represented.
        value: usize,
    },
    /// Computed Euler characteristic cannot be represented by the durable manifest.
    #[error("Euler characteristic {value} is outside the supported i64 range")]
    EulerOutOfRange {
        /// Euler characteristic that could not be represented.
        value: isize,
    },
    /// Level-3 f-vector computation failed.
    #[error("failed to compute checkpoint topology evidence: {source}")]
    TopologyComputation {
        /// Level-3 topology computation failure.
        #[source]
        source: Box<TopologyError>,
    },
    /// Toroidal period count does not match the compile-time dimension.
    #[error("toroidal topology has {actual} periods; expected {expected}")]
    ToroidalDimensionMismatch {
        /// Compile-time number of periods.
        expected: usize,
        /// Number of serialized periods.
        actual: usize,
    },
    /// Serialized toroidal domain does not satisfy domain invariants.
    #[error("invalid serialized toroidal domain: {source}")]
    ToroidalDomain {
        /// Typed toroidal-domain parsing failure.
        #[source]
        source: ToroidalDomainError,
    },
    /// Encoding the already validated TDS into its embedded checkpoint image failed.
    #[error("failed to encode the canonical checkpoint TDS image: {message}")]
    TdsSerialization {
        /// Codec diagnostic from the embedded CBOR image.
        message: String,
    },
    /// The embedded checkpoint TDS image is not valid CBOR for its raw schema.
    #[error("failed to decode the canonical checkpoint TDS image: {message}")]
    TdsCodec {
        /// Opaque diagnostic from the embedded CBOR decoder.
        message: String,
    },
    /// The decoded embedded image failed typed UUID-snapshot reconstruction.
    #[error("failed to hydrate the canonical checkpoint TDS image: {source}")]
    TdsHydration {
        /// Typed semantic snapshot reconstruction failure.
        #[source]
        source: Box<DelaunayCheckpointTdsHydrationError>,
    },
    /// The embedded image contains bytes after its single canonical TDS value.
    #[error("embedded TDS image contains {count} trailing byte(s)")]
    TrailingTdsBytes {
        /// Number of bytes remaining after the first decoded TDS value.
        count: usize,
    },
}

impl ser::Error for DelaunayCheckpointError {
    fn custom<T: fmt::Display>(message: T) -> Self {
        Self::PayloadSerialization {
            message: message.to_string(),
        }
    }
}

/// One user payload captured through Serde exactly once.
///
/// Both digest-v1 and the embedded CBOR image consume this passive value. A
/// payload with interior mutability therefore cannot make those two durable
/// representations observe different states during one checkpoint operation.
#[derive(Clone, Debug)]
struct PreparedPayload {
    value: Value,
    canonical: Vec<u8>,
}

impl PreparedPayload {
    /// Captures and validates one present payload for the digest-v1 value model.
    fn capture<T: ?Sized + Serialize>(payload: &T) -> Result<Self, DelaunayCheckpointError> {
        let canonical = canonical_fragment(payload)?;
        let value = decode_canonical_payload_value(&canonical)?;
        Ok(Self { value, canonical })
    }

    /// Reconstructs the caller's payload type from the authenticated value.
    fn deserialize<T: DataDeserialize>(&self) -> Result<T, DelaunayCheckpointError> {
        self.value
            .deserialized()
            .map_err(|source| DelaunayCheckpointError::TdsCodec {
                message: source.to_string(),
            })
    }

    /// Writes this captured value into digest-v1's canonical representation.
    fn encode<S: CanonicalSink>(&self, encoder: &mut CanonicalEncoder<S>) {
        encoder.write(&self.canonical);
    }
}

impl Serialize for PreparedPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PreparedPayload {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let mut encoder = CanonicalEncoder::fragment(Vec::new());
        encode_materialized_payload_value(&mut encoder, &value).map_err(de::Error::custom)?;
        Ok(Self {
            value,
            canonical: encoder.into_sink(),
        })
    }
}

// =============================================================================
// CANONICAL DIGEST ENCODING
// =============================================================================

/// Minimal sink required by the digest-v1 canonical encoder.
trait CanonicalSink {
    fn write(&mut self, bytes: &[u8]);
}

impl CanonicalSink for Vec<u8> {
    fn write(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

impl CanonicalSink for Sha256 {
    fn write(&mut self, bytes: &[u8]) {
        Digest::update(self, bytes);
    }
}

/// Canonical encoder that streams into its sink except for sortable map fragments.
struct CanonicalEncoder<S> {
    sink: S,
}

impl<S: CanonicalSink> CanonicalEncoder<S> {
    /// Starts a domain-separated checkpoint representation.
    fn new(mut sink: S) -> Self {
        sink.write(CANONICAL_CHECKPOINT_DOMAIN);
        Self { sink }
    }

    /// Starts a fragment without the checkpoint domain prefix.
    const fn fragment(sink: S) -> Self {
        Self { sink }
    }

    /// Finishes encoding and returns the underlying sink.
    fn into_sink(self) -> S {
        self.sink
    }

    /// Writes raw canonical bytes.
    fn write(&mut self, bytes: &[u8]) {
        self.sink.write(bytes);
    }

    /// Writes a stable one-byte type or field tag.
    fn tag(&mut self, tag: u8) {
        self.write(&[tag]);
    }

    /// Writes an unsigned integer in fixed-width network byte order.
    fn u64(&mut self, value: u64) {
        self.write(&value.to_be_bytes());
    }

    /// Writes a collection length without architecture-dependent `usize` bytes.
    fn len(&mut self, value: usize) -> Result<(), DelaunayCheckpointError> {
        let value = u64::try_from(value)
            .map_err(|_| DelaunayCheckpointError::CanonicalLengthOutOfRange { value })?;
        self.u64(value);
        Ok(())
    }

    /// Writes length-delimited bytes so adjacent values cannot be ambiguous.
    fn byte_string(&mut self, value: &[u8]) -> Result<(), DelaunayCheckpointError> {
        self.len(value.len())?;
        self.write(value);
        Ok(())
    }

    /// Writes a stable UUID as its 16 network-order bytes.
    fn uuid(&mut self, value: Uuid) {
        self.write(value.as_bytes());
    }

    /// Writes one canonical integer after the caller checked CBOR's width.
    fn integer(&mut self, value: i128) {
        self.tag(0);
        self.write(&value.to_be_bytes());
    }

    /// Writes one canonical text value.
    fn text(&mut self, value: &str) -> Result<(), DelaunayCheckpointError> {
        self.tag(3);
        self.byte_string(value.as_bytes())
    }

    /// Sorts and writes buffered map-entry fragments, rejecting duplicate keys.
    fn map_entries(
        &mut self,
        mut entries: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> Result<(), DelaunayCheckpointError> {
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(DelaunayCheckpointError::DuplicatePayloadMapKey);
        }

        self.tag(8);
        self.len(entries.len())?;
        for (key, value) in entries {
            self.byte_string(&key)?;
            self.byte_string(&value)?;
        }
        Ok(())
    }

    /// Writes a one-entry externally tagged enum map.
    fn named_value(&mut self, name: &str, value: Vec<u8>) -> Result<(), DelaunayCheckpointError> {
        self.map_entries(vec![(canonical_fragment(name)?, value)])
    }

    /// Encodes an optional payload from its single captured Serde observation.
    fn prepared_payload(&mut self, payload: Option<&PreparedPayload>) {
        if let Some(payload) = payload {
            self.tag(1);
            payload.encode(self);
        } else {
            self.tag(0);
        }
    }
}

/// Encodes one map fragment without a checkpoint domain prefix.
fn canonical_fragment<T: ?Sized + Serialize>(
    value: &T,
) -> Result<Vec<u8>, DelaunayCheckpointError> {
    let mut encoder = CanonicalEncoder::fragment(Vec::new());
    value.serialize(&mut encoder)?;
    Ok(encoder.into_sink())
}

/// Encodes one materialized Serde value with digest-v1's stable type model.
fn encode_materialized_payload_value<S: CanonicalSink>(
    encoder: &mut CanonicalEncoder<S>,
    value: &Value,
) -> Result<(), DelaunayCheckpointError> {
    match value {
        Value::Integer(value) => encoder.integer(i128::from(*value)),
        Value::Bytes(value) => {
            encoder.tag(1);
            encoder.byte_string(value)?;
        }
        Value::Float(value) => {
            encoder.tag(2);
            encoder.u64(value.to_bits());
        }
        Value::Text(value) => encoder.text(value)?,
        Value::Bool(value) => {
            encoder.tag(4);
            encoder.tag(u8::from(*value));
        }
        Value::Null => return Err(DelaunayCheckpointError::AmbiguousPayload),
        Value::Tag(tag, value) => {
            encoder.tag(6);
            encoder.u64(*tag);
            encode_materialized_payload_value(encoder, value)?;
        }
        Value::Array(values) => {
            encoder.tag(7);
            encoder.len(values.len())?;
            for value in values {
                encode_materialized_payload_value(encoder, value)?;
            }
        }
        Value::Map(values) => {
            let entries = values
                .iter()
                .map(|(key, value)| {
                    let mut key_encoder = CanonicalEncoder::fragment(Vec::new());
                    encode_materialized_payload_value(&mut key_encoder, key)?;
                    let mut value_encoder = CanonicalEncoder::fragment(Vec::new());
                    encode_materialized_payload_value(&mut value_encoder, value)?;
                    Ok((key_encoder.into_sink(), value_encoder.into_sink()))
                })
                .collect::<Result<Vec<_>, DelaunayCheckpointError>>()?;
            encoder.map_entries(entries)?;
        }
        _ => return Err(DelaunayCheckpointError::UnsupportedPayloadValue),
    }
    Ok(())
}

/// Cursor for digest-v1 payload fragments produced by [`CanonicalEncoder`].
struct CanonicalPayloadCursor<'bytes> {
    bytes: &'bytes [u8],
    position: usize,
}

impl<'bytes> CanonicalPayloadCursor<'bytes> {
    /// Creates a cursor at the start of one canonical payload fragment.
    const fn new(bytes: &'bytes [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Returns whether the complete fragment has been consumed.
    const fn is_finished(&self) -> bool {
        self.position == self.bytes.len()
    }

    /// Reads one exact byte range without trusting encoded lengths.
    fn take(&mut self, len: usize) -> Result<&'bytes [u8], DelaunayCheckpointError> {
        let end = self
            .position
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(DelaunayCheckpointError::UnsupportedPayloadValue)?;
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    /// Reads one byte.
    fn byte(&mut self) -> Result<u8, DelaunayCheckpointError> {
        Ok(self.take(1)?[0])
    }

    /// Reads one network-order `u64`.
    fn u64(&mut self) -> Result<u64, DelaunayCheckpointError> {
        let bytes = self
            .take(size_of::<u64>())?
            .try_into()
            .map_err(|_| DelaunayCheckpointError::UnsupportedPayloadValue)?;
        Ok(u64::from_be_bytes(bytes))
    }

    /// Reads one network-order `i128`.
    fn i128(&mut self) -> Result<i128, DelaunayCheckpointError> {
        let bytes = self
            .take(size_of::<i128>())?
            .try_into()
            .map_err(|_| DelaunayCheckpointError::UnsupportedPayloadValue)?;
        Ok(i128::from_be_bytes(bytes))
    }

    /// Reads one canonical collection length in the local address width.
    fn len(&mut self) -> Result<usize, DelaunayCheckpointError> {
        usize::try_from(self.u64()?).map_err(|_| DelaunayCheckpointError::UnsupportedPayloadValue)
    }

    /// Reads one length-delimited canonical fragment.
    fn fragment(&mut self) -> Result<&'bytes [u8], DelaunayCheckpointError> {
        let len = self.len()?;
        self.take(len)
    }

    /// Parses one value from digest-v1's injective Serde representation.
    fn value(&mut self) -> Result<Value, DelaunayCheckpointError> {
        match self.byte()? {
            0 => ciborium::value::Integer::try_from(self.i128()?)
                .map(Value::Integer)
                .map_err(|_| DelaunayCheckpointError::UnsupportedPayloadValue),
            1 => {
                let len = self.len()?;
                Ok(Value::Bytes(self.take(len)?.to_vec()))
            }
            2 => Ok(Value::Float(f64::from_bits(self.u64()?))),
            3 => {
                let len = self.len()?;
                let text = str::from_utf8(self.take(len)?)
                    .map_err(|_| DelaunayCheckpointError::UnsupportedPayloadValue)?;
                Ok(Value::Text(text.to_owned()))
            }
            4 => match self.byte()? {
                0 => Ok(Value::Bool(false)),
                1 => Ok(Value::Bool(true)),
                _ => Err(DelaunayCheckpointError::UnsupportedPayloadValue),
            },
            6 => {
                let tag = self.u64()?;
                Ok(Value::Tag(tag, Box::new(self.value()?)))
            }
            7 => {
                let len = self.len()?;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.value()?);
                }
                Ok(Value::Array(values))
            }
            8 => {
                let len = self.len()?;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push((
                        decode_canonical_payload_value(self.fragment()?)?,
                        decode_canonical_payload_value(self.fragment()?)?,
                    ));
                }
                Ok(Value::Map(values))
            }
            _ => Err(DelaunayCheckpointError::UnsupportedPayloadValue),
        }
    }
}

/// Converts one validated digest-v1 fragment into its passive CBOR value.
fn decode_canonical_payload_value(bytes: &[u8]) -> Result<Value, DelaunayCheckpointError> {
    let mut cursor = CanonicalPayloadCursor::new(bytes);
    let value = cursor.value()?;
    if !cursor.is_finished() {
        return Err(DelaunayCheckpointError::UnsupportedPayloadValue);
    }
    Ok(value)
}

/// Streaming sequence state for a declared-length Serde collection.
struct CanonicalSequence<'a, S> {
    encoder: &'a mut CanonicalEncoder<S>,
    expected: usize,
    actual: usize,
}

impl<S: CanonicalSink> CanonicalSequence<'_, S> {
    /// Encodes one declared sequence element and rejects writes past its promised length.
    ///
    /// Keeping the count check here gives every Serde sequence shape the same
    /// public [`DelaunayCheckpointError::PayloadCollectionLengthMismatch`] classification.
    fn element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), DelaunayCheckpointError> {
        if self.actual == self.expected {
            return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                expected: self.expected,
                actual: self.actual + 1,
            });
        }
        value.serialize(&mut *self.encoder)?;
        self.actual += 1;
        Ok(())
    }

    /// Rejects a sequence that ended before emitting its promised number of elements.
    const fn finish(self) -> Result<(), DelaunayCheckpointError> {
        if self.actual == self.expected {
            Ok(())
        } else {
            Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                expected: self.expected,
                actual: self.actual,
            })
        }
    }
}

impl<S: CanonicalSink> SerializeSeq for CanonicalSequence<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.element(value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

impl<S: CanonicalSink> SerializeTuple for CanonicalSequence<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.element(value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

impl<S: CanonicalSink> SerializeTupleStruct for CanonicalSequence<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.element(value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

/// Buffered map state; values are fragments because sorting is key-dependent.
struct CanonicalMap<'a, S> {
    encoder: &'a mut CanonicalEncoder<S>,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    pending_key: Option<Vec<u8>>,
    expected: Option<usize>,
}

impl<S: CanonicalSink> CanonicalMap<'_, S> {
    /// Buffers one named struct field using the same canonical key/value form as maps.
    fn field<T: ?Sized + Serialize>(
        &mut self,
        key: &str,
        value: &T,
    ) -> Result<(), DelaunayCheckpointError> {
        self.entries
            .push((canonical_fragment(key)?, canonical_fragment(value)?));
        Ok(())
    }

    /// Verifies map pairing and length before canonical key sorting writes the map.
    ///
    /// This is the shared boundary for incomplete-entry, length-mismatch, and
    /// duplicate-key errors exposed by checkpoint manifest construction.
    fn finish(self) -> Result<(), DelaunayCheckpointError> {
        if self.pending_key.is_some() {
            return Err(DelaunayCheckpointError::IncompletePayloadMapEntry);
        }
        if let Some(expected) = self.expected
            && self.entries.len() != expected
        {
            return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                expected,
                actual: self.entries.len(),
            });
        }
        self.encoder.map_entries(self.entries)
    }
}

impl<S: CanonicalSink> SerializeMap for CanonicalMap<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Self::Error> {
        if self.pending_key.is_some() {
            return Err(DelaunayCheckpointError::IncompletePayloadMapEntry);
        }
        self.pending_key = Some(canonical_fragment(key)?);
        Ok(())
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Self::Error> {
        let key = self
            .pending_key
            .take()
            .ok_or(DelaunayCheckpointError::IncompletePayloadMapEntry)?;
        self.entries.push((key, canonical_fragment(value)?));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

impl<S: CanonicalSink> SerializeStruct for CanonicalMap<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.field(key, value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.finish()
    }
}

/// Streaming externally tagged tuple-variant state, including CBOR tags.
enum CanonicalTupleVariant<'a, S> {
    Named {
        encoder: &'a mut CanonicalEncoder<S>,
        name: &'static str,
        value: CanonicalEncoder<Vec<u8>>,
        expected: usize,
        actual: usize,
    },
    Tagged {
        encoder: &'a mut CanonicalEncoder<S>,
        tag: Option<u64>,
        value: Option<Vec<u8>>,
        actual: usize,
    },
}

impl<S: CanonicalSink> SerializeTupleVariant for CanonicalTupleVariant<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, field: &T) -> Result<(), Self::Error> {
        match self {
            Self::Named {
                value,
                expected,
                actual,
                ..
            } => {
                if *actual == *expected {
                    return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                        expected: *expected,
                        actual: *actual + 1,
                    });
                }
                field.serialize(&mut *value)?;
                *actual += 1;
            }
            Self::Tagged {
                tag, value, actual, ..
            } => {
                match *actual {
                    0 => {
                        let encoded = canonical_fragment(field)?;
                        let bytes: [u8; 16] = encoded
                            .get(1..)
                            .and_then(|bytes| bytes.try_into().ok())
                            .filter(|_| encoded.first() == Some(&0))
                            .ok_or(DelaunayCheckpointError::InvalidPayloadCborTag)?;
                        *tag = Some(
                            u64::try_from(i128::from_be_bytes(bytes))
                                .map_err(|_| DelaunayCheckpointError::InvalidPayloadCborTag)?,
                        );
                    }
                    1 => *value = Some(canonical_fragment(field)?),
                    _ => {
                        return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                            expected: 2,
                            actual: *actual + 1,
                        });
                    }
                }
                *actual += 1;
            }
        }
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        match self {
            Self::Named {
                encoder,
                name,
                value,
                expected,
                actual,
            } => {
                if actual != expected {
                    return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                        expected,
                        actual,
                    });
                }
                encoder.named_value(name, value.into_sink())
            }
            Self::Tagged {
                encoder,
                tag,
                value,
                actual,
            } => {
                if actual != 2 {
                    return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                        expected: 2,
                        actual,
                    });
                }
                encoder.tag(6);
                encoder.u64(tag.ok_or(DelaunayCheckpointError::InvalidPayloadCborTag)?);
                encoder.write(&value.ok_or(DelaunayCheckpointError::InvalidPayloadCborTag)?);
                Ok(())
            }
        }
    }
}

/// Buffered struct-variant fields forming the value of an outer enum map.
struct CanonicalStructVariant<'a, S> {
    encoder: &'a mut CanonicalEncoder<S>,
    name: &'static str,
    entries: Vec<(Vec<u8>, Vec<u8>)>,
    expected: usize,
}

impl<S: CanonicalSink> SerializeStructVariant for CanonicalStructVariant<'_, S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.entries
            .push((canonical_fragment(key)?, canonical_fragment(value)?));
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        if self.entries.len() != self.expected {
            return Err(DelaunayCheckpointError::PayloadCollectionLengthMismatch {
                expected: self.expected,
                actual: self.entries.len(),
            });
        }
        let mut value = CanonicalEncoder::fragment(Vec::new());
        value.map_entries(self.entries)?;
        self.encoder.named_value(self.name, value.into_sink())
    }
}

impl<'a, S: CanonicalSink> Serializer for &'a mut CanonicalEncoder<S> {
    type Ok = ();
    type Error = DelaunayCheckpointError;
    type SerializeSeq = CanonicalSequence<'a, S>;
    type SerializeTuple = CanonicalSequence<'a, S>;
    type SerializeTupleStruct = CanonicalSequence<'a, S>;
    type SerializeTupleVariant = CanonicalTupleVariant<'a, S>;
    type SerializeMap = CanonicalMap<'a, S>;
    type SerializeStruct = CanonicalMap<'a, S>;
    type SerializeStructVariant = CanonicalStructVariant<'a, S>;

    fn serialize_bool(self, value: bool) -> Result<Self::Ok, Self::Error> {
        self.tag(4);
        self.tag(u8::from(value));
        Ok(())
    }

    fn serialize_i8(self, value: i8) -> Result<Self::Ok, Self::Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i16(self, value: i16) -> Result<Self::Ok, Self::Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i32(self, value: i32) -> Result<Self::Ok, Self::Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i64(self, value: i64) -> Result<Self::Ok, Self::Error> {
        self.serialize_i128(i128::from(value))
    }

    fn serialize_i128(self, value: i128) -> Result<Self::Ok, Self::Error> {
        let cbor_magnitude = if value.is_negative() {
            value ^ !0
        } else {
            value
        };
        u64::try_from(cbor_magnitude)
            .map_err(|_| DelaunayCheckpointError::SignedPayloadIntegerOutOfRange { value })?;
        self.integer(value);
        Ok(())
    }

    fn serialize_u8(self, value: u8) -> Result<Self::Ok, Self::Error> {
        self.serialize_u128(u128::from(value))
    }

    fn serialize_u16(self, value: u16) -> Result<Self::Ok, Self::Error> {
        self.serialize_u128(u128::from(value))
    }

    fn serialize_u32(self, value: u32) -> Result<Self::Ok, Self::Error> {
        self.serialize_u128(u128::from(value))
    }

    fn serialize_u64(self, value: u64) -> Result<Self::Ok, Self::Error> {
        self.serialize_u128(u128::from(value))
    }

    fn serialize_u128(self, value: u128) -> Result<Self::Ok, Self::Error> {
        let value = u64::try_from(value)
            .map_err(|_| DelaunayCheckpointError::UnsignedPayloadIntegerOutOfRange { value })?;
        self.integer(i128::from(value));
        Ok(())
    }

    fn serialize_f32(self, value: f32) -> Result<Self::Ok, Self::Error> {
        self.serialize_f64(f64::from(value))
    }

    fn serialize_f64(self, value: f64) -> Result<Self::Ok, Self::Error> {
        self.tag(2);
        self.u64(value.to_bits());
        Ok(())
    }

    fn serialize_char(self, value: char) -> Result<Self::Ok, Self::Error> {
        let mut bytes = [0; 4];
        self.serialize_str(value.encode_utf8(&mut bytes))
    }

    fn serialize_str(self, value: &str) -> Result<Self::Ok, Self::Error> {
        self.text(value)
    }

    fn serialize_bytes(self, value: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.tag(1);
        self.byte_string(value)
    }

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        Err(DelaunayCheckpointError::AmbiguousPayload)
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        Err(DelaunayCheckpointError::AmbiguousPayload)
    }

    fn serialize_unit_struct(self, _name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.serialize_str(variant)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        if (name, variant) == ("@@TAG@@", "@@UNTAGGED@@") {
            value.serialize(self)
        } else {
            self.named_value(variant, canonical_fragment(value)?)
        }
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        let expected = len.ok_or(DelaunayCheckpointError::UnboundedPayloadSequence)?;
        self.tag(7);
        self.len(expected)?;
        Ok(CanonicalSequence {
            encoder: self,
            expected,
            actual: 0,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        if (name, variant) == ("@@TAG@@", "@@TAGGED@@") {
            Ok(CanonicalTupleVariant::Tagged {
                encoder: self,
                tag: None,
                value: None,
                actual: 0,
            })
        } else {
            let mut value = CanonicalEncoder::fragment(Vec::new());
            value.tag(7);
            value.len(len)?;
            Ok(CanonicalTupleVariant::Named {
                encoder: self,
                name: variant,
                value,
                expected: len,
                actual: 0,
            })
        }
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Ok(CanonicalMap {
            encoder: self,
            entries: Vec::with_capacity(len.unwrap_or(0)),
            pending_key: None,
            expected: len,
        })
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(CanonicalMap {
            encoder: self,
            entries: Vec::with_capacity(len),
            pending_key: None,
            expected: Some(len),
        })
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Ok(CanonicalStructVariant {
            encoder: self,
            name: variant,
            entries: Vec::with_capacity(len),
            expected: len,
        })
    }

    fn is_human_readable(&self) -> bool {
        false
    }
}

/// Encodes proof context in fixed field order for digest-v1 checkpoints.
fn encode_proof_context<S: CanonicalSink, const D: usize>(
    encoder: &mut CanonicalEncoder<S>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<(), DelaunayCheckpointError> {
    encoder.tag(match topology_guarantee {
        TopologyGuaranteeWire::Pseudomanifold => 0,
        TopologyGuaranteeWire::PlManifold => 1,
    });

    match global_topology {
        GlobalTopologyWire::Euclidean {} => encoder.tag(0),
        GlobalTopologyWire::Toroidal { periods, mode } => {
            encoder.tag(1);
            encoder.len(periods.0.len())?;
            for &period_bits in &periods.0 {
                encoder.u64(period_bits);
            }
            encoder.tag(match mode {
                ToroidalConstructionModeWire::PeriodicImagePoint => 0,
                ToroidalConstructionModeWire::Explicit => 1,
            });
        }
        GlobalTopologyWire::Spherical {} => encoder.tag(2),
        GlobalTopologyWire::Hyperbolic {} => encoder.tag(3),
    }

    encoder.tag(match validation_policy {
        ValidationPolicyWire::Never => 0,
        ValidationPolicyWire::ExplicitOnly => 1,
        ValidationPolicyWire::OnSuspicion => 2,
        ValidationPolicyWire::Always => 3,
        ValidationPolicyWire::DebugOnly => 4,
    });
    Ok(())
}

/// Computes digest-v1 from the exact payload representation stored in the image.
fn digest_prepared_tds<const D: usize>(
    snapshot: &TdsSnapshot<PreparedPayload, PreparedPayload, D>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<String, DelaunayCheckpointError> {
    let mut encoder = CanonicalEncoder::new(Sha256::new());
    encoder.write(&DELAUNAY_CHECKPOINT_SCHEMA_VERSION.to_be_bytes());
    encoder.write(&DELAUNAY_CHECKPOINT_DIGEST_VERSION.to_be_bytes());
    encoder.len(D)?;
    encode_proof_context(
        &mut encoder,
        topology_guarantee,
        global_topology,
        validation_policy,
    )?;

    let mut vertices = snapshot.vertices().collect::<Vec<_>>();
    vertices.sort_unstable_by_key(|vertex| vertex.uuid());
    encoder.len(vertices.len())?;
    for vertex in vertices {
        encoder.uuid(vertex.uuid());
        for coordinate in vertex.point().coords() {
            encoder.u64(coordinate.to_bits());
        }
        encoder.prepared_payload(vertex.data());
    }

    let mut simplices = snapshot.simplices().collect::<Vec<_>>();
    simplices.sort_unstable_by_key(|simplex| simplex.uuid());
    encoder.len(simplices.len())?;
    for simplex in simplices {
        let simplex_uuid = simplex.uuid();
        encoder.uuid(simplex_uuid);
        encoder.prepared_payload(simplex.data());

        encoder.len(simplex.vertex_uuids().len())?;
        for &vertex_uuid in simplex.vertex_uuids() {
            encoder.uuid(vertex_uuid);
        }

        let neighbors = simplex.neighbor_uuids();
        encoder.len(neighbors.len())?;
        for neighbor_uuid in neighbors {
            match neighbor_uuid {
                Some(neighbor_uuid) => {
                    encoder.tag(1);
                    encoder.uuid(*neighbor_uuid);
                }
                None => encoder.tag(0),
            }
        }

        match simplex.periodic_vertex_offsets() {
            Some(offsets) => {
                encoder.tag(1);
                encoder.len(offsets.len())?;
                for offset in offsets {
                    for component in offset {
                        encoder.write(&component.to_be_bytes());
                    }
                }
            }
            None => encoder.tag(0),
        }
    }

    let digest = encoder.into_sink().finalize();
    Ok(lower_hex(&digest))
}

/// Formats bytes as lowercase hexadecimal without a second formatting dependency.
fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

/// Converts the authoritative Level-3 f-vector into the durable manifest width.
fn durable_f_vector(values: &[usize]) -> Result<Vec<u64>, DelaunayCheckpointError> {
    values
        .iter()
        .copied()
        .map(|value| {
            u64::try_from(value)
                .map_err(|_| DelaunayCheckpointError::FVectorEntryOutOfRange { value })
        })
        .collect()
}

/// Converts the Level-3 Euler implementation's result into the durable manifest width.
fn durable_euler(value: isize) -> Result<i64, DelaunayCheckpointError> {
    i64::try_from(value).map_err(|_| DelaunayCheckpointError::EulerOutOfRange { value })
}

/// Establishes Levels 1–2 evidence bound to one immutable TDS borrow.
fn prepare_checkpoint_tds<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<ValidatedTdsSerialization<'_, U, V, D>, DelaunayCheckpointError> {
    ValidatedTdsSerialization::try_new(tds).map_err(|source| {
        DelaunayCheckpointError::TdsValidation {
            source: Box::new(source),
        }
    })
}

/// Captures every present user payload once in the validated UUID snapshot.
fn prepare_checkpoint_payloads<U, V, const D: usize>(
    validated: &ValidatedTdsSerialization<'_, U, V, D>,
) -> Result<TdsSnapshot<PreparedPayload, PreparedPayload, D>, DelaunayCheckpointError>
where
    U: DataSerialize,
    V: DataSerialize,
{
    let snapshot =
        validated
            .snapshot()
            .map_err(|source| DelaunayCheckpointError::TdsSerialization {
                message: source.to_string(),
            })?;
    snapshot.try_map_payloads(
        |payload| PreparedPayload::capture(*payload),
        |payload| PreparedPayload::capture(*payload),
    )
}

/// Computes the manifest from Levels 1–2 evidence for the unchanged owner.
fn build_manifest_from_validated<U, V, const D: usize>(
    validated: &ValidatedTdsSerialization<'_, U, V, D>,
    prepared: &TdsSnapshot<PreparedPayload, PreparedPayload, D>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<DelaunayCheckpointManifest, DelaunayCheckpointError>
where
    U: DataSerialize,
    V: DataSerialize,
{
    let tds = validated.tds();
    let dimension = u32::try_from(D)
        .map_err(|_| DelaunayCheckpointError::DimensionOutOfRange { dimension: D })?;
    let counts =
        count_simplices(tds).map_err(|source| DelaunayCheckpointError::TopologyComputation {
            source: Box::new(source),
        })?;
    let euler = durable_euler(euler_characteristic(&counts))?;
    let f_vector = durable_f_vector(&counts.by_dim)?;
    let digest = digest_prepared_tds(
        prepared,
        topology_guarantee,
        global_topology,
        validation_policy,
    )?;

    Ok(DelaunayCheckpointManifest {
        manifest_version: DELAUNAY_CHECKPOINT_MANIFEST_VERSION,
        dimension,
        f_vector,
        euler_characteristic: euler,
        digest: DelaunayCheckpointDigest {
            version: DELAUNAY_CHECKPOINT_DIGEST_VERSION,
            algorithm: DelaunayCheckpointDigestAlgorithm::Sha256,
            value: digest,
        },
    })
}

/// Validates one TDS and computes its scientific manifest.
fn build_manifest<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<DelaunayCheckpointManifest, DelaunayCheckpointError>
where
    U: DataSerialize,
    V: DataSerialize,
{
    let validated = prepare_checkpoint_tds(tds)?;
    let prepared = prepare_checkpoint_payloads(&validated)?;
    build_manifest_from_validated(
        &validated,
        &prepared,
        topology_guarantee,
        global_topology,
        validation_policy,
    )
}

/// Computes an alternating sum from untrusted manifest counts with checked width.
fn manifest_euler(f_vector: &[u64]) -> Result<i64, DelaunayCheckpointError> {
    let derived = f_vector
        .iter()
        .copied()
        .enumerate()
        .try_fold(0_i128, |sum, (dimension, count)| {
            let count = i128::from(count);
            if dimension % 2 == 0 {
                sum.checked_add(count)
            } else {
                sum.checked_sub(count)
            }
        })
        .ok_or(DelaunayCheckpointError::ManifestEulerOutOfRange)?;
    i64::try_from(derived).map_err(|_| DelaunayCheckpointError::ManifestEulerOutOfRange)
}

/// Manifest whose version, dimension, topology metrics, and digest shape agree.
#[derive(Clone, Copy, Debug)]
struct ShapeCheckedManifest<'manifest, const D: usize> {
    manifest: &'manifest DelaunayCheckpointManifest,
    _dimension: PhantomData<[(); D]>,
}

impl<'manifest, const D: usize> ShapeCheckedManifest<'manifest, D> {
    const fn manifest(self) -> &'manifest DelaunayCheckpointManifest {
        self.manifest
    }
}

/// Verifies manifest-only shape and consistency before consulting owner state.
fn verify_manifest_shape<const D: usize>(
    manifest: &DelaunayCheckpointManifest,
) -> Result<ShapeCheckedManifest<'_, D>, DelaunayCheckpointError> {
    if manifest.manifest_version != DELAUNAY_CHECKPOINT_MANIFEST_VERSION {
        return Err(DelaunayCheckpointError::UnsupportedManifestVersion {
            actual: manifest.manifest_version,
            expected: DELAUNAY_CHECKPOINT_MANIFEST_VERSION,
        });
    }
    let expected_dimension = u32::try_from(D)
        .map_err(|_| DelaunayCheckpointError::DimensionOutOfRange { dimension: D })?;
    if manifest.dimension != expected_dimension {
        return Err(DelaunayCheckpointError::DimensionMismatch {
            expected: expected_dimension,
            actual: manifest.dimension,
        });
    }
    let expected_len = D + 1;
    if manifest.f_vector.len() != expected_len {
        return Err(DelaunayCheckpointError::FVectorLengthMismatch {
            expected: expected_len,
            actual: manifest.f_vector.len(),
        });
    }
    let derived = manifest_euler(&manifest.f_vector)?;
    if manifest.euler_characteristic != derived {
        return Err(DelaunayCheckpointError::ManifestEulerMismatch {
            declared: manifest.euler_characteristic,
            derived,
        });
    }
    if manifest.digest.version != DELAUNAY_CHECKPOINT_DIGEST_VERSION {
        return Err(DelaunayCheckpointError::UnsupportedDigestVersion {
            actual: manifest.digest.version,
            expected: DELAUNAY_CHECKPOINT_DIGEST_VERSION,
        });
    }
    if manifest.digest.algorithm != DelaunayCheckpointDigestAlgorithm::Sha256 {
        return Err(DelaunayCheckpointError::UnsupportedDigestAlgorithm {
            actual: manifest.digest.algorithm.clone(),
            expected: DelaunayCheckpointDigestAlgorithm::Sha256,
        });
    }
    if manifest.digest.value.len() != SHA256_HEX_LENGTH
        || !manifest
            .digest
            .value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DelaunayCheckpointError::MalformedDigest {
            actual: manifest.digest.value.clone(),
        });
    }
    Ok(ShapeCheckedManifest {
        manifest,
        _dimension: PhantomData,
    })
}

/// Verifies the canonical digest against the exact embedded payload representation.
fn verify_manifest_digest_against_snapshot<const D: usize>(
    checked_manifest: ShapeCheckedManifest<'_, D>,
    snapshot: &TdsSnapshot<PreparedPayload, PreparedPayload, D>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<(), DelaunayCheckpointError> {
    let manifest = checked_manifest.manifest();
    let computed_digest = digest_prepared_tds(
        snapshot,
        topology_guarantee,
        global_topology,
        validation_policy,
    )?;
    if manifest.digest.value != computed_digest {
        return Err(DelaunayCheckpointError::DigestMismatch {
            declared: manifest.digest.value.clone(),
            computed: computed_digest,
        });
    }

    Ok(())
}

/// Compares a manifest with already computed Level-3 metrics.
fn verify_manifest_topology_metrics<const D: usize>(
    checked_manifest: ShapeCheckedManifest<'_, D>,
    computed_f_vector: Vec<u64>,
    computed_euler: i64,
) -> Result<(), DelaunayCheckpointError> {
    let manifest = checked_manifest.manifest();
    if manifest.f_vector != computed_f_vector {
        return Err(DelaunayCheckpointError::FVectorMismatch {
            declared: manifest.f_vector.clone(),
            computed: computed_f_vector,
        });
    }
    if manifest.euler_characteristic != computed_euler {
        return Err(DelaunayCheckpointError::EulerCharacteristicMismatch {
            declared: manifest.euler_characteristic,
            computed: computed_euler,
        });
    }
    Ok(())
}

/// Verifies digest and recomputed topology evidence without treating either as proof.
fn verify_manifest_against_tds<U, V, const D: usize>(
    checked_manifest: ShapeCheckedManifest<'_, D>,
    tds: &Tds<U, V, D>,
    prepared: &TdsSnapshot<PreparedPayload, PreparedPayload, D>,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: &GlobalTopologyWire<D>,
    validation_policy: ValidationPolicyWire,
) -> Result<(), DelaunayCheckpointError>
where
    U: DataSerialize,
    V: DataSerialize,
{
    verify_manifest_digest_against_snapshot(
        checked_manifest,
        prepared,
        topology_guarantee,
        global_topology,
        validation_policy,
    )?;
    let counts =
        count_simplices(tds).map_err(|source| DelaunayCheckpointError::TopologyComputation {
            source: Box::new(source),
        })?;
    verify_manifest_topology_metrics(
        checked_manifest,
        durable_f_vector(&counts.by_dim)?,
        durable_euler(euler_characteristic(&counts))?,
    )
}

/// Consumes metrics from the exact Level-3 restoration pass.
fn verify_manifest_with_evidence<U, V, const D: usize>(
    checked_manifest: ShapeCheckedManifest<'_, D>,
    tds: &Tds<U, V, D>,
    evidence: &TopologyCertificationEvidence,
) -> Result<(), DelaunayCheckpointError> {
    let counts = evidence
        .simplex_counts(tds)
        .ok_or(DelaunayCheckpointError::StaleTopologyEvidence)?;
    let euler = evidence
        .euler_characteristic(tds)
        .ok_or(DelaunayCheckpointError::StaleTopologyEvidence)?;
    verify_manifest_topology_metrics(
        checked_manifest,
        durable_f_vector(&counts.by_dim)?,
        durable_euler(euler)?,
    )
}

/// Encodes the same prepared snapshot used to construct digest-v1.
fn encode_prepared_tds<U, V, const D: usize>(
    prepared: TdsSnapshot<U, V, D>,
) -> Result<TdsBytes, DelaunayCheckpointError>
where
    U: DataSerialize,
    V: DataSerialize,
{
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&prepared.into_raw(), &mut bytes).map_err(|source| {
        DelaunayCheckpointError::TdsSerialization {
            message: source.to_string(),
        }
    })?;
    Ok(TdsBytes(bytes))
}

/// Decodes and validates the embedded UUID snapshot without hydrating payload types.
fn decode_prepared_tds<const D: usize>(
    bytes: &TdsBytes,
) -> Result<TdsSnapshot<PreparedPayload, PreparedPayload, D>, DelaunayCheckpointError> {
    let mut remaining = bytes.0.as_slice();
    let snapshot: RawTdsSnapshot<PreparedPayload, PreparedPayload, D> =
        ciborium::de::from_reader(&mut remaining).map_err(|source| {
            DelaunayCheckpointError::TdsCodec {
                message: source.to_string(),
            }
        })?;
    if !remaining.is_empty() {
        return Err(DelaunayCheckpointError::TrailingTdsBytes {
            count: remaining.len(),
        });
    }
    snapshot
        .parse()
        .map_err(|source| DelaunayCheckpointError::TdsHydration {
            source: Box::new(source.into()),
        })
}

/// Hydrates authenticated payload values and publishes one validated runtime TDS.
fn hydrate_prepared_tds<U, V, const D: usize>(
    prepared: &TdsSnapshot<PreparedPayload, PreparedPayload, D>,
) -> Result<Tds<U, V, D>, DelaunayCheckpointError>
where
    U: DataDeserialize,
    V: DataDeserialize,
{
    prepared
        .try_map_payloads(PreparedPayload::deserialize, PreparedPayload::deserialize)?
        .into_tds()
        .map_err(|source| DelaunayCheckpointError::TdsHydration {
            source: Box::new(source.into()),
        })
}

// =============================================================================
// PUBLIC OWNER CHECKPOINT API
// =============================================================================

/// Typed failure while validating, hydrating, or restoring a decoded checkpoint.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum DelaunayCheckpointLoadError {
    /// Checkpoint metadata, digest, payload, or TDS-image validation failed.
    #[error(transparent)]
    Checkpoint {
        /// Typed checkpoint failure.
        #[from]
        source: DelaunayCheckpointError,
    },
    /// Topology guarantee and validation cadence are incompatible.
    #[error(transparent)]
    ValidationConfiguration {
        /// Typed validation-configuration failure.
        #[from]
        source: ValidationConfigurationError,
    },
    /// Independent high-dimensional construction replay failed.
    #[error("high-dimensional checkpoint provenance replay failed: {source}")]
    ProvenanceReplay {
        /// Trusted construction failure.
        #[source]
        source: Box<DelaunayTriangulationConstructionError>,
    },
    /// The trusted replay produced different sorted vertex signatures.
    #[error(
        "high-dimensional checkpoint vertex provenance differs at sorted index {index}: checkpoint UUID {checkpoint_vertex_uuid:?} with coordinate bits {checkpoint_coordinate_bits:?}, replayed UUID {replayed_vertex_uuid:?} with coordinate bits {replayed_coordinate_bits:?} (checkpoint count {checkpoint_count}, replayed count {replayed_count})"
    )]
    ProvenanceVertexMismatch {
        /// First sorted signature index that differs or is missing on one side.
        index: usize,
        /// Vertex UUID stored at the differing checkpoint position, if present.
        checkpoint_vertex_uuid: Option<Uuid>,
        /// Coordinate bit patterns stored at the differing checkpoint position.
        checkpoint_coordinate_bits: Option<Vec<u64>>,
        /// Vertex UUID produced at the differing replay position, if present.
        replayed_vertex_uuid: Option<Uuid>,
        /// Coordinate bit patterns produced at the differing replay position.
        replayed_coordinate_bits: Option<Vec<u64>>,
        /// Number of checkpoint vertex signatures.
        checkpoint_count: usize,
        /// Number of replayed vertex signatures.
        replayed_count: usize,
    },
    /// The trusted replay produced different sorted maximal-simplex signatures.
    #[error(
        "high-dimensional checkpoint simplex provenance differs at sorted index {index}: checkpoint vertices {checkpoint_vertices:?}, replayed vertices {replayed_vertices:?} (checkpoint count {checkpoint_count}, replayed count {replayed_count})"
    )]
    ProvenanceSimplexMismatch {
        /// First sorted simplex-signature index that differs or is missing on one side.
        index: usize,
        /// Sorted checkpoint vertex UUIDs at the differing position, if present.
        checkpoint_vertices: Option<Vec<Uuid>>,
        /// Sorted replayed vertex UUIDs at the differing position, if present.
        replayed_vertices: Option<Vec<Uuid>>,
        /// Number of checkpoint maximal-simplex signatures.
        checkpoint_count: usize,
        /// Number of replayed maximal-simplex signatures.
        replayed_count: usize,
    },
    /// The crate cannot currently reconstruct trusted periodic provenance in this dimension.
    #[error(
        "periodic PL-manifold checkpoint restoration is unsupported in dimension {dimension}; trusted periodic construction currently supports dimensions at most 3"
    )]
    UnsupportedPeriodicProvenance {
        /// Compile-time dimension requiring unavailable proof reconstruction.
        dimension: usize,
    },
    /// Levels 3–4 restoration failed with a typed validation reason.
    #[error("checkpoint TDS-to-triangulation restoration failed: {source}")]
    TriangulationRestoration {
        /// Typed Levels 3–4 failure.
        #[source]
        source: Box<TriangulationBuilderError>,
    },
    /// Level 5 restoration failed with a typed validation reason.
    #[error("checkpoint triangulation-to-Delaunay restoration failed: {source}")]
    DelaunayRestoration {
        /// Typed Level 5 failure.
        #[source]
        source: Box<DelaunayTriangulationValidationError>,
    },
}

/// Codec-stage failure returned by [`DelaunayCheckpoint::decode`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DelaunayCheckpointDecodeError<E> {
    /// The outer serde codec rejected the raw bounded checkpoint envelope.
    Codec {
        /// Original codec error, preserved without string conversion.
        source: E,
    },
}

impl<E: fmt::Display> fmt::Display for DelaunayCheckpointDecodeError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec { source } => write!(formatter, "checkpoint codec failed: {source}"),
        }
    }
}

impl<E: StdError + 'static> StdError for DelaunayCheckpointDecodeError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Codec { source } => Some(source),
        }
    }
}

/// Raw, codec-decoded schema-v2 checkpoint awaiting typed validation and load.
///
/// Deserialize this type when callers need to inspect typed checkpoint or
/// restoration failures. Calling [`Self::try_into_delaunay_with_kernel`]
/// validates schema, manifest shape, and proof context before hydrating TDS
/// storage. It then verifies the semantic digest before replaying provenance
/// and re-establishing Levels 3–5 with the supplied kernel.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::checkpoint::{
///     DELAUNAY_CHECKPOINT_SCHEMA_VERSION, DelaunayCheckpoint,
///     DelaunayCheckpointLoadError,
/// };
/// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// #     #[error(transparent)]
/// #     Json(#[from] serde_json::Error),
/// #     #[error(transparent)]
/// #     Load(#[from] DelaunayCheckpointLoadError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// let vertices = vec![
///     vertex![0.0, 0.0]?,
///     vertex![1.0, 0.0]?,
///     vertex![0.0, 1.0]?,
/// ];
/// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// let json = serde_json::to_string(&triangulation)?;
/// let checkpoint: DelaunayCheckpoint<(), (), 2> = serde_json::from_str(&json)?;
///
/// assert_eq!(checkpoint.schema_version(), DELAUNAY_CHECKPOINT_SCHEMA_VERSION);
/// assert!(checkpoint.manifest().is_some());
/// let restored = checkpoint.try_into_delaunay()?;
/// assert_eq!(restored.number_of_simplices(), triangulation.number_of_simplices());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct DelaunayCheckpoint<U, V, const D: usize> {
    wire: DelaunayTriangulationWire<D>,
    payload: PhantomData<fn() -> (U, V)>,
}

/// Explicit compatibility loader for manifest-less schema-v1 owner snapshots.
///
/// Schema v1 cannot provide integrity evidence and its outer floating-point
/// codec may already have lost information. This type therefore exists only as
/// an executable migration bridge: decode with the codec used by the legacy
/// artifact, re-establish Levels 1–5, then serialize the returned owner to emit
/// schema v2. Ordinary schema-v2 loading never falls back to this format.
///
/// # Examples
///
/// ```no_run
/// use delaunay::prelude::checkpoint::{
///     DelaunayCheckpointLoadError, DelaunayCheckpointV1,
/// };
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum MigrationError {
/// #     #[error(transparent)]
/// #     Io(#[from] std::io::Error),
/// #     #[error(transparent)]
/// #     Json(#[from] serde_json::Error),
/// #     #[error(transparent)]
/// #     Load(#[from] DelaunayCheckpointLoadError),
/// # }
/// # fn main() -> Result<(), MigrationError> {
/// let legacy_json = std::fs::read_to_string("checkpoint-v1.json")?;
/// let checkpoint: DelaunayCheckpointV1<(), (), 3> =
///     serde_json::from_str(&legacy_json)?;
/// let triangulation = checkpoint.try_into_delaunay()?;
///
/// let schema_v2_json = serde_json::to_string_pretty(&triangulation)?;
/// std::fs::write("checkpoint-v2.json", schema_v2_json)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct DelaunayCheckpointV1<U, V, const D: usize> {
    wire: DelaunayTriangulationV1Wire<Tds<U, V, D>, D>,
}

impl<'de, U, V, const D: usize> Deserialize<'de> for DelaunayCheckpointV1<U, V, D>
where
    U: DataDeserialize,
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(Self {
            wire: DelaunayTriangulationV1Wire::deserialize(deserializer)?,
        })
    }
}

impl<U, V, const D: usize> DelaunayCheckpointV1<U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Restores a legacy checkpoint with the default robust kernel.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointLoadError::Checkpoint`] when the schema marker or
    ///   serialized topology metadata is invalid;
    /// - [`DelaunayCheckpointLoadError::ValidationConfiguration`] when the
    ///   stored topology guarantee and validation policy are incompatible;
    /// - [`DelaunayCheckpointLoadError::ProvenanceReplay`],
    ///   [`DelaunayCheckpointLoadError::ProvenanceVertexMismatch`], or
    ///   [`DelaunayCheckpointLoadError::ProvenanceSimplexMismatch`] when an
    ///   independent high-dimensional Euclidean reconstruction fails or
    ///   disagrees;
    /// - [`DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance`] when a
    ///   periodic owner requires unavailable high-dimensional provenance;
    /// - [`DelaunayCheckpointLoadError::TriangulationRestoration`] when Levels
    ///   3–4 cannot be restored; or
    /// - [`DelaunayCheckpointLoadError::DelaunayRestoration`] when Level 5 fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::DelaunayTriangulation;
    /// use delaunay::prelude::checkpoint::{
    ///     DelaunayCheckpointLoadError, DelaunayCheckpointV1,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// fn restore_legacy(
    ///     checkpoint: DelaunayCheckpointV1<(), (), 3>,
    /// ) -> Result<
    ///     DelaunayTriangulation<RobustKernel<f64>, (), (), 3>,
    ///     DelaunayCheckpointLoadError,
    /// > {
    ///     checkpoint.try_into_delaunay()
    /// }
    /// ```
    pub fn try_into_delaunay(
        self,
    ) -> Result<DelaunayTriangulation<RobustKernel<f64>, U, V, D>, DelaunayCheckpointLoadError>
    where
        RobustKernel<f64>: ExactPredicates<D>,
    {
        self.try_into_delaunay_with_kernel(RobustKernel::new())
    }

    /// Restores a legacy checkpoint with a caller-supplied exact kernel.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointLoadError::Checkpoint`] when the schema marker or
    ///   serialized topology metadata is invalid;
    /// - [`DelaunayCheckpointLoadError::ValidationConfiguration`] when the
    ///   stored topology guarantee and validation policy are incompatible;
    /// - [`DelaunayCheckpointLoadError::ProvenanceReplay`],
    ///   [`DelaunayCheckpointLoadError::ProvenanceVertexMismatch`], or
    ///   [`DelaunayCheckpointLoadError::ProvenanceSimplexMismatch`] when an
    ///   independent high-dimensional Euclidean reconstruction fails or
    ///   disagrees;
    /// - [`DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance`] when a
    ///   periodic owner requires unavailable high-dimensional provenance;
    /// - [`DelaunayCheckpointLoadError::TriangulationRestoration`] when Levels
    ///   3–4 cannot be restored; or
    /// - [`DelaunayCheckpointLoadError::DelaunayRestoration`] when Level 5 fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::DelaunayTriangulation;
    /// use delaunay::prelude::checkpoint::{
    ///     DelaunayCheckpointLoadError, DelaunayCheckpointV1,
    /// };
    /// use delaunay::prelude::geometry::AdaptiveKernel;
    ///
    /// fn restore_legacy_with_adaptive_kernel(
    ///     checkpoint: DelaunayCheckpointV1<(), (), 3>,
    /// ) -> Result<
    ///     DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3>,
    ///     DelaunayCheckpointLoadError,
    /// > {
    ///     checkpoint.try_into_delaunay_with_kernel(AdaptiveKernel::new())
    /// }
    /// ```
    pub fn try_into_delaunay_with_kernel<K>(
        self,
        kernel: K,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayCheckpointLoadError>
    where
        K: ExactPredicates<D>,
    {
        let DelaunayTriangulationV1Wire {
            schema_version,
            tds,
            topology_guarantee,
            global_topology,
            validation_policy,
        } = self.wire;
        if schema_version != 1 {
            return Err(DelaunayCheckpointError::UnsupportedSchemaVersion {
                actual: schema_version,
                expected: 1,
            }
            .into());
        }
        let topology_guarantee: TopologyGuarantee = topology_guarantee.into();
        let validation_policy: ValidationPolicy = validation_policy.into();
        if !topology_guarantee.is_compatible_with_policy(validation_policy) {
            return Err(
                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                    topology_guarantee,
                    validation_policy,
                }
                .into(),
            );
        }
        let global_topology = global_topology.try_into_global_topology()?;
        let provenance =
            replay_checkpoint_provenance(&tds, &kernel, topology_guarantee, global_topology)?;
        let (mut triangulation, _topology_evidence) = restore_tds_with_provenance(
            tds,
            kernel,
            topology_guarantee,
            global_topology,
            provenance,
        )?;
        triangulation.try_set_validation_policy(validation_policy)?;
        Ok(triangulation)
    }
}

impl<'de, U, V, const D: usize> Deserialize<'de> for DelaunayCheckpoint<U, V, D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        Ok(Self {
            wire: DelaunayTriangulationWire::deserialize(deserializer)?,
            payload: PhantomData,
        })
    }
}

impl<U, V, const D: usize> DelaunayCheckpoint<U, V, D> {
    /// Decodes the raw bounded envelope while preserving the outer codec error.
    ///
    /// Scientific and proof validation is deliberately deferred to
    /// [`Self::try_into_delaunay_with_kernel`], whose error remains fully typed.
    ///
    /// # Errors
    ///
    /// Returns [`DelaunayCheckpointDecodeError::Codec`] when the outer format or
    /// a bounded fixed-arity field is malformed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::checkpoint::{
    ///     DELAUNAY_CHECKPOINT_SCHEMA_VERSION, DelaunayCheckpoint,
    ///     DelaunayCheckpointDecodeError,
    /// };
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #     #[error(transparent)]
    /// #     Json(#[from] serde_json::Error),
    /// #     #[error(transparent)]
    /// #     Decode(#[from] DelaunayCheckpointDecodeError<serde_json::Error>),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let json = serde_json::to_string(&triangulation)?;
    /// let mut deserializer = serde_json::Deserializer::from_str(&json);
    /// let checkpoint = DelaunayCheckpoint::<(), (), 2>::decode(&mut deserializer)?;
    ///
    /// assert_eq!(checkpoint.schema_version(), DELAUNAY_CHECKPOINT_SCHEMA_VERSION);
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode<'de, De>(
        deserializer: De,
    ) -> Result<Self, DelaunayCheckpointDecodeError<De::Error>>
    where
        De: Deserializer<'de>,
    {
        DelaunayTriangulationWire::deserialize(deserializer)
            .map(|wire| Self {
                wire,
                payload: PhantomData,
            })
            .map_err(|source| DelaunayCheckpointDecodeError::Codec { source })
    }

    /// Returns the untrusted owner schema version from the decoded envelope.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::checkpoint::{
    ///     DELAUNAY_CHECKPOINT_SCHEMA_VERSION, DelaunayCheckpoint,
    /// };
    ///
    /// fn is_current_schema(checkpoint: &DelaunayCheckpoint<(), (), 2>) -> bool {
    ///     checkpoint.schema_version() == DELAUNAY_CHECKPOINT_SCHEMA_VERSION
    /// }
    /// ```
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.wire.schema_version
    }

    /// Borrows the untrusted scientific manifest when present.
    ///
    /// The manifest remains owned by the decoded checkpoint. Clone it explicitly
    /// only when it must outlive or be retained after consuming the checkpoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::checkpoint::DelaunayCheckpoint;
    ///
    /// fn checkpoint_dimension(checkpoint: &DelaunayCheckpoint<(), (), 2>) -> Option<u32> {
    ///     checkpoint.manifest().map(|manifest| manifest.dimension)
    /// }
    /// ```
    #[must_use]
    pub fn manifest(&self) -> Option<&DelaunayCheckpointManifest> {
        self.wire.manifest.as_ref().map(|manifest| &manifest.0)
    }
}

impl<U, V, const D: usize> DelaunayCheckpoint<U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Restores the checkpoint with the default robust exact-predicate kernel.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointLoadError::Checkpoint`] when metadata, manifest
    ///   evidence, canonical payloads, or the embedded TDS image are invalid;
    /// - [`DelaunayCheckpointLoadError::ValidationConfiguration`] when the
    ///   stored topology guarantee and validation policy are incompatible;
    /// - [`DelaunayCheckpointLoadError::ProvenanceReplay`],
    ///   [`DelaunayCheckpointLoadError::ProvenanceVertexMismatch`], or
    ///   [`DelaunayCheckpointLoadError::ProvenanceSimplexMismatch`] when an
    ///   independent high-dimensional Euclidean reconstruction fails or
    ///   disagrees;
    /// - [`DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance`] when a
    ///   periodic owner requires unavailable high-dimensional provenance;
    /// - [`DelaunayCheckpointLoadError::TriangulationRestoration`] when Levels
    ///   3–4 cannot be restored; or
    /// - [`DelaunayCheckpointLoadError::DelaunayRestoration`] when Level 5 fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::DelaunayTriangulation;
    /// use delaunay::prelude::checkpoint::{
    ///     DelaunayCheckpoint, DelaunayCheckpointLoadError,
    /// };
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// fn restore(
    ///     checkpoint: DelaunayCheckpoint<(), (), 2>,
    /// ) -> Result<
    ///     DelaunayTriangulation<RobustKernel<f64>, (), (), 2>,
    ///     DelaunayCheckpointLoadError,
    /// > {
    ///     checkpoint.try_into_delaunay()
    /// }
    /// ```
    pub fn try_into_delaunay(
        self,
    ) -> Result<DelaunayTriangulation<RobustKernel<f64>, U, V, D>, DelaunayCheckpointLoadError>
    where
        RobustKernel<f64>: ExactPredicates<D>,
    {
        self.try_into_delaunay_with_kernel(RobustKernel::new())
    }

    /// Restores the checkpoint with a caller-supplied exact-predicate kernel.
    ///
    /// The checkpoint's topology guarantee, global topology, and validation
    /// policy are restored together. High-dimensional PL-manifold provenance is
    /// never trusted from the wire: Euclidean owners are independently rebuilt
    /// and compared before proof is reattached; unsupported periodic cases are
    /// rejected explicitly.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointLoadError::Checkpoint`] when metadata, manifest
    ///   evidence, canonical payloads, or the embedded TDS image are invalid;
    /// - [`DelaunayCheckpointLoadError::ValidationConfiguration`] when the
    ///   stored topology guarantee and validation policy are incompatible;
    /// - [`DelaunayCheckpointLoadError::ProvenanceReplay`],
    ///   [`DelaunayCheckpointLoadError::ProvenanceVertexMismatch`], or
    ///   [`DelaunayCheckpointLoadError::ProvenanceSimplexMismatch`] when an
    ///   independent high-dimensional Euclidean reconstruction fails or
    ///   disagrees;
    /// - [`DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance`] when a
    ///   periodic owner requires unavailable high-dimensional provenance;
    /// - [`DelaunayCheckpointLoadError::TriangulationRestoration`] when Levels
    ///   3–4 cannot be restored; or
    /// - [`DelaunayCheckpointLoadError::DelaunayRestoration`] when Level 5 fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::DelaunayTriangulation;
    /// use delaunay::prelude::checkpoint::{
    ///     DelaunayCheckpoint, DelaunayCheckpointLoadError,
    /// };
    /// use delaunay::prelude::geometry::AdaptiveKernel;
    ///
    /// fn restore_with_adaptive_kernel(
    ///     checkpoint: DelaunayCheckpoint<(), (), 2>,
    /// ) -> Result<
    ///     DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2>,
    ///     DelaunayCheckpointLoadError,
    /// > {
    ///     checkpoint.try_into_delaunay_with_kernel(AdaptiveKernel::new())
    /// }
    /// ```
    pub fn try_into_delaunay_with_kernel<K>(
        self,
        kernel: K,
    ) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayCheckpointLoadError>
    where
        K: ExactPredicates<D>,
    {
        restore_checkpoint(self.wire, kernel)
    }
}

/// Returns UUID/coordinate signatures in stable order for replay comparison.
fn vertex_signatures<U, V, const D: usize>(tds: &Tds<U, V, D>) -> Vec<(Uuid, [u64; D])> {
    let mut signatures = tds
        .vertices()
        .map(|(_, vertex)| {
            (
                vertex.uuid(),
                std::array::from_fn(|axis| vertex.point().coords()[axis].to_bits()),
            )
        })
        .collect::<Vec<_>>();
    signatures.sort_unstable_by_key(|(uuid, _)| *uuid);
    signatures
}

/// Returns maximal simplices as sorted vertex-UUID multisets.
fn euclidean_simplex_signatures<U, V, const D: usize>(
    tds: &Tds<U, V, D>,
) -> Result<Vec<Vec<Uuid>>, DelaunayCheckpointLoadError> {
    let mut signatures = Vec::with_capacity(tds.number_of_simplices());
    for (_, simplex) in tds.simplices() {
        let mut vertices = simplex
            .vertices()
            .iter()
            .map(|&vertex_key| {
                tds.vertex_uuid_from_key(vertex_key).ok_or_else(|| {
                    DelaunayCheckpointError::MissingVertexReference {
                        simplex_uuid: simplex.uuid(),
                        vertex_key,
                    }
                    .into()
                })
            })
            .collect::<Result<Vec<_>, DelaunayCheckpointLoadError>>()?;
        vertices.sort_unstable();
        signatures.push(vertices);
    }
    signatures.sort_unstable();
    Ok(signatures)
}

/// Returns the first unequal or one-sided position in two sorted signature slices.
fn first_signature_mismatch<T: PartialEq>(left: &[T], right: &[T]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

/// Verifies independently produced vertex and maximal-simplex signatures.
fn verify_provenance_signatures<const D: usize>(
    checkpoint_vertices: &[(Uuid, [u64; D])],
    replayed_vertices: &[(Uuid, [u64; D])],
    checkpoint_simplices: &[Vec<Uuid>],
    replayed_simplices: &[Vec<Uuid>],
) -> Result<(), DelaunayCheckpointLoadError> {
    if let Some(index) = first_signature_mismatch(checkpoint_vertices, replayed_vertices) {
        let (checkpoint_vertex_uuid, checkpoint_coordinate_bits) = checkpoint_vertices
            .get(index)
            .map_or((None, None), |(uuid, coordinates)| {
                (Some(*uuid), Some(coordinates.to_vec()))
            });
        let (replayed_vertex_uuid, replayed_coordinate_bits) = replayed_vertices
            .get(index)
            .map_or((None, None), |(uuid, coordinates)| {
                (Some(*uuid), Some(coordinates.to_vec()))
            });
        return Err(DelaunayCheckpointLoadError::ProvenanceVertexMismatch {
            index,
            checkpoint_vertex_uuid,
            checkpoint_coordinate_bits,
            replayed_vertex_uuid,
            replayed_coordinate_bits,
            checkpoint_count: checkpoint_vertices.len(),
            replayed_count: replayed_vertices.len(),
        });
    }

    if let Some(index) = first_signature_mismatch(checkpoint_simplices, replayed_simplices) {
        return Err(DelaunayCheckpointLoadError::ProvenanceSimplexMismatch {
            index,
            checkpoint_vertices: checkpoint_simplices.get(index).cloned(),
            replayed_vertices: replayed_simplices.get(index).cloned(),
            checkpoint_count: checkpoint_simplices.len(),
            replayed_count: replayed_simplices.len(),
        });
    }

    Ok(())
}

/// Reconstructs only provenance that can be proven by an independent replay.
fn replay_checkpoint_provenance<K, U, V, const D: usize>(
    tds: &Tds<U, V, D>,
    kernel: &K,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
) -> Result<TopologyConstructionProvenance, DelaunayCheckpointLoadError>
where
    K: ExactPredicates<D>,
{
    if D < 4
        || topology_guarantee != TopologyGuarantee::PLManifold
        || tds.number_of_simplices() <= 1
    {
        return Ok(TopologyConstructionProvenance::Unproven);
    }
    if global_topology.is_toroidal() {
        return Err(DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance { dimension: D });
    }
    if !global_topology.is_euclidean() {
        return Ok(TopologyConstructionProvenance::Unproven);
    }

    let mut vertices = tds
        .vertices()
        .map(|(_, vertex)| {
            Vertex::from_validated_point_with_uuid(*vertex.point(), vertex.uuid(), None)
        })
        .collect::<Vec<Vertex<(), D>>>();
    vertices.sort_unstable_by_key(Vertex::uuid);
    let replay: DelaunayTriangulation<K, (), (), D> = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(topology_guarantee)
        .build_with_kernel(kernel)
        .map_err(|source| DelaunayCheckpointLoadError::ProvenanceReplay {
            source: Box::new(source),
        })?;

    let checkpoint_vertices = vertex_signatures(tds);
    let replayed_vertices = vertex_signatures(&replay.tri.tds);
    let checkpoint_simplices = euclidean_simplex_signatures(tds)?;
    let replayed_simplices = euclidean_simplex_signatures(&replay.tri.tds)?;
    verify_provenance_signatures(
        &checkpoint_vertices,
        &replayed_vertices,
        &checkpoint_simplices,
        &replayed_simplices,
    )?;
    Ok(TopologyConstructionProvenance::EuclideanDelaunayInsertion)
}

/// Drops a retained lower-layer owner while preserving its typed failure reason.
fn map_restoration_error<K, U, V, const D: usize>(
    failure: DelaunayTdsRestorationError<K, U, V, D>,
) -> DelaunayCheckpointLoadError {
    match failure.into_reason() {
        DelaunayTdsRestorationReason::Triangulation { source } => {
            DelaunayCheckpointLoadError::TriangulationRestoration {
                source: Box::new(source),
            }
        }
        DelaunayTdsRestorationReason::Delaunay { source } => {
            DelaunayCheckpointLoadError::DelaunayRestoration {
                source: Box::new(source),
            }
        }
    }
}

/// Restores through one Level-3 pass and retains its owner-bound metrics.
fn restore_tds_with_provenance<K, U, V, const D: usize>(
    tds: Tds<U, V, D>,
    kernel: K,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    provenance: TopologyConstructionProvenance,
) -> Result<
    (
        DelaunayTriangulation<K, U, V, D>,
        TopologyCertificationEvidence,
    ),
    DelaunayCheckpointLoadError,
>
where
    K: Kernel<D, Scalar = f64>,
    U: DataType,
    V: DataType,
{
    DelaunayTriangulation::try_restore_with_provenance(
        tds,
        kernel,
        topology_guarantee,
        global_topology,
        provenance,
    )
    .map_err(map_restoration_error)
}

/// Validates one raw envelope, hydrates its TDS, and restores Levels 1–5.
fn restore_checkpoint<K, U, V, const D: usize>(
    wire: DelaunayTriangulationWire<D>,
    kernel: K,
) -> Result<DelaunayTriangulation<K, U, V, D>, DelaunayCheckpointLoadError>
where
    K: ExactPredicates<D>,
    U: DataType,
    V: DataType,
{
    if wire.schema_version != DELAUNAY_CHECKPOINT_SCHEMA_VERSION {
        return Err(DelaunayCheckpointError::UnsupportedSchemaVersion {
            actual: wire.schema_version,
            expected: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
        }
        .into());
    }
    let manifest = wire
        .manifest
        .ok_or(DelaunayCheckpointError::MissingManifest)?
        .0;
    let checked_manifest = verify_manifest_shape::<D>(&manifest)?;

    let topology_guarantee: TopologyGuarantee = wire.topology_guarantee.into();
    let validation_policy: ValidationPolicy = wire.validation_policy.into();
    if !topology_guarantee.is_compatible_with_policy(validation_policy) {
        return Err(
            ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                topology_guarantee,
                validation_policy,
            }
            .into(),
        );
    }
    let prepared = decode_prepared_tds(&wire.tds)?;
    verify_manifest_digest_against_snapshot(
        checked_manifest,
        &prepared,
        wire.topology_guarantee,
        &wire.global_topology,
        wire.validation_policy,
    )?;
    let tds = hydrate_prepared_tds(&prepared)?;
    let global_topology = wire.global_topology.try_into_global_topology()?;
    let provenance =
        replay_checkpoint_provenance(&tds, &kernel, topology_guarantee, global_topology)?;

    let (mut triangulation, topology_evidence) =
        restore_tds_with_provenance(tds, kernel, topology_guarantee, global_topology, provenance)?;
    verify_manifest_with_evidence(checked_manifest, &triangulation.tri.tds, &topology_evidence)?;
    triangulation.try_set_validation_policy(validation_policy)?;
    Ok(triangulation)
}

impl<K, U, V, const D: usize> DelaunayTriangulation<K, U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    /// Recomputes the scientific manifest that serialization will embed.
    ///
    /// This is a derived snapshot, not a mutable cache and not a replacement
    /// for [`Self::validate`]. A valid bistellar move can change its f-vector
    /// and digest while preserving its Euler characteristic.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointError::TdsValidation`] when Levels 1–2 fail;
    /// - [`DelaunayCheckpointError::TopologyComputation`] when the f-vector
    ///   cannot be computed;
    /// - [`DelaunayCheckpointError::DimensionOutOfRange`],
    ///   [`DelaunayCheckpointError::FVectorEntryOutOfRange`],
    ///   [`DelaunayCheckpointError::EulerOutOfRange`], or
    ///   [`DelaunayCheckpointError::CanonicalLengthOutOfRange`] when a value
    ///   exceeds its durable width;
    /// - [`DelaunayCheckpointError::PayloadSerialization`],
    ///   [`DelaunayCheckpointError::SignedPayloadIntegerOutOfRange`],
    ///   [`DelaunayCheckpointError::UnsignedPayloadIntegerOutOfRange`],
    ///   [`DelaunayCheckpointError::InvalidPayloadCborTag`],
    ///   [`DelaunayCheckpointError::AmbiguousPayload`],
    ///   [`DelaunayCheckpointError::DuplicatePayloadMapKey`],
    ///   [`DelaunayCheckpointError::UnboundedPayloadSequence`],
    ///   [`DelaunayCheckpointError::PayloadCollectionLengthMismatch`],
    ///   [`DelaunayCheckpointError::IncompletePayloadMapEntry`], or
    ///   [`DelaunayCheckpointError::UnsupportedPayloadValue`] when a user
    ///   payload cannot be represented injectively by digest-v1; or
    /// - [`DelaunayCheckpointError::MissingVertexReference`],
    ///   [`DelaunayCheckpointError::MissingNeighborSlots`], or
    ///   [`DelaunayCheckpointError::MissingNeighborReference`] when canonical
    ///   TDS relationships cannot be resolved.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::checkpoint::DELAUNAY_CHECKPOINT_SCHEMA_VERSION;
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Checkpoint(#[from] delaunay::checkpoint::DelaunayCheckpointError),
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    ///
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let manifest = triangulation.checkpoint_manifest()?;
    ///
    /// assert_eq!(DELAUNAY_CHECKPOINT_SCHEMA_VERSION, 2);
    /// assert_eq!(manifest.dimension, 2);
    /// assert_eq!(manifest.f_vector, vec![3, 3, 1]);
    /// assert_eq!(manifest.euler_characteristic, 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn checkpoint_manifest(
        &self,
    ) -> Result<DelaunayCheckpointManifest, DelaunayCheckpointError> {
        let topology_guarantee = self.topology_guarantee().into();
        let global_topology = self.global_topology().into();
        let validation_policy = self.validation_policy().into();
        build_manifest(
            &self.tri.tds,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
    }

    /// Verifies untrusted manifest evidence against this owner state.
    ///
    /// This returns typed shape, digest, f-vector, and Euler mismatch errors for
    /// downstream checkpoint tooling. It does not reconstruct topology and is
    /// not a substitute for [`Self::validate`]; the serde load boundary always
    /// performs this evidence check before re-establishing Levels 3–5.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    /// - [`DelaunayCheckpointError::UnsupportedManifestVersion`],
    ///   [`DelaunayCheckpointError::DimensionMismatch`],
    ///   [`DelaunayCheckpointError::FVectorLengthMismatch`],
    ///   [`DelaunayCheckpointError::ManifestEulerMismatch`],
    ///   [`DelaunayCheckpointError::ManifestEulerOutOfRange`],
    ///   [`DelaunayCheckpointError::UnsupportedDigestVersion`],
    ///   [`DelaunayCheckpointError::UnsupportedDigestAlgorithm`], or
    ///   [`DelaunayCheckpointError::MalformedDigest`] when the manifest is
    ///   malformed or internally inconsistent;
    /// - [`DelaunayCheckpointError::TdsValidation`] or
    ///   [`DelaunayCheckpointError::TopologyComputation`] when current owner
    ///   evidence cannot be recomputed;
    /// - [`DelaunayCheckpointError::DimensionOutOfRange`],
    ///   [`DelaunayCheckpointError::FVectorEntryOutOfRange`],
    ///   [`DelaunayCheckpointError::EulerOutOfRange`], or
    ///   [`DelaunayCheckpointError::CanonicalLengthOutOfRange`] when a value
    ///   exceeds its durable width;
    /// - [`DelaunayCheckpointError::PayloadSerialization`],
    ///   [`DelaunayCheckpointError::SignedPayloadIntegerOutOfRange`],
    ///   [`DelaunayCheckpointError::UnsignedPayloadIntegerOutOfRange`],
    ///   [`DelaunayCheckpointError::InvalidPayloadCborTag`],
    ///   [`DelaunayCheckpointError::AmbiguousPayload`],
    ///   [`DelaunayCheckpointError::DuplicatePayloadMapKey`],
    ///   [`DelaunayCheckpointError::UnboundedPayloadSequence`],
    ///   [`DelaunayCheckpointError::PayloadCollectionLengthMismatch`],
    ///   [`DelaunayCheckpointError::IncompletePayloadMapEntry`], or
    ///   [`DelaunayCheckpointError::UnsupportedPayloadValue`] when a user
    ///   payload cannot be represented injectively by digest-v1;
    /// - [`DelaunayCheckpointError::MissingVertexReference`],
    ///   [`DelaunayCheckpointError::MissingNeighborSlots`], or
    ///   [`DelaunayCheckpointError::MissingNeighborReference`] when canonical
    ///   TDS relationships cannot be resolved; or
    /// - [`DelaunayCheckpointError::DigestMismatch`],
    ///   [`DelaunayCheckpointError::FVectorMismatch`], or
    ///   [`DelaunayCheckpointError::EulerCharacteristicMismatch`] when the
    ///   manifest disagrees with recomputed owner state.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    ///
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #     #[error(transparent)]
    /// #     Checkpoint(#[from] delaunay::checkpoint::DelaunayCheckpointError),
    /// #     #[error(transparent)]
    /// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #     #[error(transparent)]
    /// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = vec![
    ///     vertex![0.0, 0.0]?,
    ///     vertex![1.0, 0.0]?,
    ///     vertex![0.0, 1.0]?,
    /// ];
    /// let triangulation = DelaunayTriangulationBuilder::new(&vertices).build()?;
    /// let manifest = triangulation.checkpoint_manifest()?;
    ///
    /// triangulation.verify_checkpoint_manifest(&manifest)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn verify_checkpoint_manifest(
        &self,
        manifest: &DelaunayCheckpointManifest,
    ) -> Result<(), DelaunayCheckpointError> {
        let checked_manifest = verify_manifest_shape::<D>(manifest)?;
        let topology_guarantee = self.topology_guarantee().into();
        let global_topology = self.global_topology().into();
        let validation_policy = self.validation_policy().into();
        let validated = prepare_checkpoint_tds(&self.tri.tds)?;
        let prepared = prepare_checkpoint_payloads(&validated)?;
        verify_manifest_against_tds(
            checked_manifest,
            validated.tds(),
            &prepared,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
    }
}

impl<K, U, V, const D: usize> Serialize for DelaunayTriangulation<K, U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let topology_guarantee = self.topology_guarantee().into();
        let global_topology = self.global_topology().into();
        let validation_policy = self.validation_policy().into();
        let validated = prepare_checkpoint_tds(&self.tri.tds).map_err(ser::Error::custom)?;
        let prepared = prepare_checkpoint_payloads(&validated).map_err(ser::Error::custom)?;
        let manifest = build_manifest_from_validated(
            &validated,
            &prepared,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
        .map_err(ser::Error::custom)?;
        let tds = encode_prepared_tds(prepared).map_err(ser::Error::custom)?;

        DelaunayTriangulationSerializeWire {
            schema_version: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
            manifest: manifest.into(),
            tds,
            topology_guarantee,
            global_topology,
            validation_policy,
        }
        .serialize(serializer)
    }
}

/// Custom `Deserialize` implementation for [`RobustKernel<f64>`].
///
/// Kernels are stateless and can be reconstructed on deserialization. The
/// checkpoint retains the canonical [`Tds`] together with the topology
/// guarantee, global topology model, and validation cadence owned by the
/// higher proof layers. Deserialization parses that passive data and then
/// composes the ordinary TDS-to-triangulation and triangulation-to-Delaunay
/// refinement boundaries, so a decoded value has re-established Levels 1–5.
///
/// Cached reports and insertion hints are deliberately not trusted across the
/// persistence boundary.
///
/// # Note on Locate Hint Persistence
///
/// The internal `insertion_state.last_inserted_simplex` locate hint is not
/// serialized. Deserialization reconstructs a fresh triangulation via
/// the same strict TDS → `Triangulation` → `DelaunayTriangulation` proof chain,
/// which resets the hint to `None`.
/// This only affects performance for the first few insertions after loading.
///
/// # Usage with Other Kernels
///
/// The direct serde boundary reconstructs [`RobustKernel<f64>`] because a
/// caller-supplied kernel cannot be obtained from `Deserialize`. Decode a
/// [`DelaunayCheckpoint`] and use its typed custom-kernel load boundary instead;
/// it restores the TDS, topology guarantee, global topology, and validation
/// policy together before re-establishing Levels 1–5.
///
/// ```rust
/// # use delaunay::prelude::checkpoint::{DelaunayCheckpoint, DelaunayCheckpointLoadError};
/// # use delaunay::prelude::construction::DelaunayTriangulationBuilder;
/// # use delaunay::prelude::geometry::AdaptiveKernel;
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Serde(#[from] serde_json::Error),
/// #     #[error(transparent)]
/// #     Load(#[from] DelaunayCheckpointLoadError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn example() -> Result<(), ExampleError> {
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// let json = serde_json::to_string(&dt)?;
/// let checkpoint: DelaunayCheckpoint<(), (), 3> = serde_json::from_str(&json)?;
/// let dt_adaptive = checkpoint.try_into_delaunay_with_kernel(AdaptiveKernel::new())?;
/// assert_eq!(dt_adaptive.number_of_vertices(), dt.number_of_vertices());
/// assert_eq!(dt_adaptive.number_of_simplices(), dt.number_of_simplices());
/// assert_eq!(dt_adaptive.topology_guarantee(), dt.topology_guarantee());
/// assert_eq!(dt_adaptive.validation_policy(), dt.validation_policy());
/// assert_eq!(dt_adaptive.global_topology(), dt.global_topology());
/// # Ok(())
/// # }
/// ```
impl<'de, U, V, const D: usize> Deserialize<'de>
    for DelaunayTriangulation<RobustKernel<f64>, U, V, D>
where
    U: DataType,
    V: DataType,
    RobustKernel<f64>: ExactPredicates<D>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        DelaunayCheckpoint::<U, V, D>::deserialize(deserializer)?
            .try_into_delaunay()
            .map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::DelaunayInsertionState;
    use crate::core::simplex::Simplex;
    #[cfg(feature = "count-allocations")]
    use crate::core::util::measure_with_result;
    use crate::delaunay_model::EuclideanDelaunayReportDomain;
    use crate::delaunayize::DelaunayRefinementBuilder;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::triangulation::Triangulation;
    use crate::triangulation::flips::BistellarFlips;
    use crate::vertex;
    use ciborium::value::Value;
    use serde::ser::{SerializeMap, SerializeTupleVariant};
    use std::{cell::Cell, collections::BTreeMap, sync::Once};

    /// Establishes the proof before encoding a test fixture.
    fn encode_tds_for_test<U, V, const D: usize>(
        tds: &Tds<U, V, D>,
    ) -> Result<TdsBytes, DelaunayCheckpointError>
    where
        U: DataSerialize,
        V: DataSerialize,
    {
        let validated = prepare_checkpoint_tds(tds)?;
        let prepared = prepare_checkpoint_payloads(&validated)?;
        encode_prepared_tds(prepared)
    }

    /// Encodes one present payload through the original streaming digest path.
    fn streaming_payload_bytes<T: DataSerialize>(
        payload: &T,
    ) -> Result<Vec<u8>, DelaunayCheckpointError> {
        let mut encoder = CanonicalEncoder::new(Vec::new());
        encoder.tag(1);
        payload.serialize(&mut encoder)?;
        Ok(encoder.into_sink())
    }

    /// Decodes and hydrates an embedded TDS for focused codec tests.
    fn decode_tds<U, V, const D: usize>(
        bytes: &TdsBytes,
    ) -> Result<Tds<U, V, D>, DelaunayCheckpointError>
    where
        U: DataDeserialize,
        V: DataDeserialize,
    {
        let prepared = decode_prepared_tds(bytes)?;
        hydrate_prepared_tds(&prepared)
    }

    struct NotAKernel;

    #[derive(Debug, Deserialize, Serialize)]
    struct OwnedSerdePayload(String);

    #[derive(Clone, Debug)]
    struct NonExactKernel(RobustKernel<f64>);

    impl Kernel<2> for NonExactKernel {
        type Scalar = f64;

        fn orientation(&self, points: &[Point<2>]) -> Result<i32, CoordinateConversionError> {
            self.0.orientation(points)
        }

        fn in_sphere(
            &self,
            simplex_points: &[Point<2>],
            test_point: &Point<2>,
        ) -> Result<i32, CoordinateConversionError> {
            self.0.in_sphere(simplex_points, test_point)
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
    struct RejectingPayload;

    impl<'de> Deserialize<'de> for RejectingPayload {
        fn deserialize<De>(_deserializer: De) -> Result<Self, De::Error>
        where
            De: Deserializer<'de>,
        {
            Err(de::Error::custom("payload hydration must not run"))
        }
    }

    #[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
    struct FailingSerializePayload;

    impl Serialize for FailingSerializePayload {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            Err(serde::ser::Error::custom("payload sentinel"))
        }
    }

    std::thread_local! {
        static STATEFUL_PAYLOAD_SERIALIZATIONS: Cell<u64> = const { Cell::new(0) };
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    struct StatefulSerializePayload;

    impl Serialize for StatefulSerializePayload {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            STATEFUL_PAYLOAD_SERIALIZATIONS.with(|serializations| {
                let observation = serializations.get();
                serializations.set(observation + 1);
                serializer.serialize_u64(observation)
            })
        }
    }

    impl<'de> Deserialize<'de> for StatefulSerializePayload {
        fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
        where
            De: Deserializer<'de>,
        {
            let _observation = u64::deserialize(deserializer)?;
            Ok(Self)
        }
    }

    struct InvalidCborTagPayload;

    impl Serialize for InvalidCborTagPayload {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut tag = serializer.serialize_tuple_variant("@@TAG@@", 0, "@@TAGGED@@", 2)?;
            tag.serialize_field("not an unsigned tag")?;
            tag.serialize_field(&7_u8)?;
            tag.end()
        }
    }

    #[derive(Serialize)]
    struct SerializeOnlyPayload(String);

    struct MapOrderPayload {
        reverse: bool,
    }

    impl Serialize for MapOrderPayload {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut map = serializer.serialize_map(Some(2))?;
            if self.reverse {
                map.serialize_entry("second", &2_u32)?;
                map.serialize_entry("first", &1_u32)?;
            } else {
                map.serialize_entry("first", &1_u32)?;
                map.serialize_entry("second", &2_u32)?;
            }
            map.end()
        }
    }

    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    struct DuplicateKeyPayload {
        reverse: bool,
    }

    impl Serialize for DuplicateKeyPayload {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mut map = serializer.serialize_map(Some(2))?;
            if self.reverse {
                map.serialize_entry("value", &2_u32)?;
                map.serialize_entry("value", &1_u32)?;
            } else {
                map.serialize_entry("value", &1_u32)?;
                map.serialize_entry("value", &2_u32)?;
            }
            map.end()
        }
    }

    impl<'de> Deserialize<'de> for DuplicateKeyPayload {
        fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
        where
            De: Deserializer<'de>,
        {
            struct DuplicateKeyVisitor;

            impl<'de> Visitor<'de> for DuplicateKeyVisitor {
                type Value = DuplicateKeyPayload;

                fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str("a duplicate-key payload map")
                }

                fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
                where
                    A: de::MapAccess<'de>,
                {
                    let mut last = None;
                    while let Some((_key, value)) = map.next_entry::<String, u32>()? {
                        last = Some(value);
                    }
                    Ok(DuplicateKeyPayload {
                        reverse: last == Some(1),
                    })
                }
            }

            deserializer.deserialize_map(DuplicateKeyVisitor)
        }
    }

    #[derive(Serialize)]
    enum SerdeShapePayload {
        Unit,
        Newtype(u32),
        Tuple(u8, bool),
        Struct { count: u16, enabled: bool },
    }

    #[derive(Serialize)]
    struct CompositeSerdePayload {
        signed: i64,
        unsigned: u64,
        float: f32,
        character: char,
        sequence: [u8; 3],
        optional: Option<u16>,
    }

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer()
                .try_init();
        });
    }

    fn non_delaunay_quad_tds() -> Tds<(), (), 2> {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([4.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([4.0, 2.0]).unwrap())
            .unwrap();
        let v3 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 2.0]).unwrap())
            .unwrap();

        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
        )
        .unwrap();
        tds.insert_simplex_with_mapping(
            Simplex::try_new_with_data(vec![v0, v2, v3], None).unwrap(),
        )
        .unwrap();
        tds.force_construction_complete_for_test();
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds
    }

    /// Builds a small payload-free owner used by manifest tamper tests.
    fn triangle_checkpoint() -> serde_json::Value {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::builder(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        serde_json::to_value(triangulation).unwrap()
    }

    /// Returns the typed schema-v2 load error for one corrupted 2D checkpoint.
    fn typed_checkpoint_error(checkpoint: serde_json::Value) -> DelaunayCheckpointLoadError {
        typed_checkpoint_error_with_payload::<(), ()>(checkpoint)
    }

    /// Returns a typed load error for a corrupted 2D checkpoint with payloads.
    fn typed_checkpoint_error_with_payload<U, V>(
        checkpoint: serde_json::Value,
    ) -> DelaunayCheckpointLoadError
    where
        U: DataType,
        V: DataType,
    {
        serde_json::from_value::<DelaunayCheckpoint<U, V, 2>>(checkpoint)
            .expect("metadata mutation should retain a decodable bounded envelope")
            .try_into_delaunay()
            .expect_err("corrupted checkpoint must be rejected")
    }

    /// Mutates the embedded CBOR TDS through a JSON-shaped test value.
    fn mutate_embedded_tds(checkpoint: &mut serde_json::Value, mutate: impl FnOnce(&mut Value)) {
        let bytes = checkpoint["tds"]
            .as_array()
            .expect("JSON checkpoint should encode TDS bytes as an array")
            .iter()
            .map(|value| {
                u8::try_from(value.as_u64().expect("TDS byte should be unsigned"))
                    .expect("TDS byte should fit u8")
            })
            .collect::<Vec<_>>();
        let mut tds: Value = ciborium::de::from_reader(bytes.as_slice())
            .expect("embedded TDS should decode for test mutation");
        mutate(&mut tds);
        let mut encoded = Vec::new();
        ciborium::ser::into_writer(&tds, &mut encoded)
            .expect("mutated TDS test value should re-encode");
        checkpoint["tds"] = serde_json::json!(encoded);
    }

    /// Returns one named field in a CBOR map used by the embedded TDS image.
    fn cbor_field_mut<'a>(value: &'a mut Value, name: &str) -> &'a mut Value {
        value
            .as_map_mut()
            .expect("embedded TDS record should be a map")
            .iter_mut()
            .find_map(|(key, value)| (key.as_text() == Some(name)).then_some(value))
            .unwrap_or_else(|| panic!("embedded TDS record should contain {name}"))
    }

    /// Returns digest-v1 payload bytes through the former materialized path.
    fn materialized_payload_bytes<T: Serialize>(
        payload: &T,
    ) -> Result<Vec<u8>, DelaunayCheckpointError> {
        let value = Value::serialized(payload).map_err(|source| {
            DelaunayCheckpointError::PayloadSerialization {
                message: source.to_string(),
            }
        })?;
        let mut encoder = CanonicalEncoder::new(Vec::new());
        encoder.tag(1);
        encode_materialized_payload_value(&mut encoder, &value)?;
        Ok(encoder.into_sink())
    }

    /// Confirms that one supported Serde shape retained its digest-v1 bytes.
    fn assert_streaming_payload_compatibility<T: DataSerialize>(payload: &T) {
        let expected = streaming_payload_bytes(payload).unwrap();
        assert_eq!(expected, materialized_payload_bytes(payload).unwrap());

        let prepared = PreparedPayload::capture(payload).unwrap();
        let mut prepared_encoder = CanonicalEncoder::new(Vec::new());
        prepared_encoder.prepared_payload(Some(&prepared));
        assert_eq!(expected, prepared_encoder.into_sink());

        let mut cbor = Vec::new();
        ciborium::ser::into_writer(&prepared, &mut cbor).unwrap();
        let decoded: PreparedPayload = ciborium::de::from_reader(cbor.as_slice()).unwrap();
        let mut decoded_encoder = CanonicalEncoder::new(Vec::new());
        decoded_encoder.prepared_payload(Some(&decoded));
        assert_eq!(expected, decoded_encoder.into_sink());
    }

    /// Builds the deterministic periodic T^2 fixture used for offset integrity coverage.
    fn periodic_checkpoint_fixture() -> DelaunayTriangulation<RobustKernel<f64>, (), (), 2> {
        let vertices = (0_u32..7)
            .map(|index| {
                let index = f64::from(index);
                vertex!([
                    0.9_f64.mul_add(((index + 1.0) * 0.618_033_988_749_894_8).fract(), 0.05,),
                    0.9_f64.mul_add(((index + 1.0) * 0.414_213_562_373_095_03).fract(), 0.05,),
                ])
                .unwrap()
            })
            .collect::<Vec<_>>();

        DelaunayTriangulation::builder(&vertices)
            .try_toroidal([1.0; 2])
            .unwrap()
            .build_with_kernel(&RobustKernel::new())
            .unwrap()
    }

    #[test]
    fn typed_checkpoint_api_preserves_domain_errors_and_custom_kernel_context() {
        let json = serde_json::to_string(&triangle_checkpoint()).unwrap();
        let decoded: DelaunayCheckpoint<(), (), 2> = serde_json::from_str(&json).unwrap();
        let restored = decoded
            .try_into_delaunay_with_kernel(AdaptiveKernel::new())
            .unwrap();
        restored.validate().unwrap();

        let mut invalid_schema = triangle_checkpoint();
        invalid_schema["schema_version"] = serde_json::json!(99);
        let decoded: DelaunayCheckpoint<RejectingPayload, (), 2> =
            serde_json::from_value(invalid_schema).unwrap();
        let error = decoded
            .try_into_delaunay_with_kernel(AdaptiveKernel::new())
            .expect_err("schema validation must precede payload hydration");
        assert!(matches!(
            error,
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::UnsupportedSchemaVersion {
                    actual: 99,
                    expected: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
                },
            }
        ));
    }

    #[test]
    fn typed_manifest_errors_and_bounded_shape_failures_are_distinct() {
        let mut unknown_algorithm = triangle_checkpoint();
        unknown_algorithm["manifest"]["digest"]["algorithm"] = serde_json::json!("sha3-256-future");
        let decoded: DelaunayCheckpoint<(), (), 2> =
            serde_json::from_value(unknown_algorithm).unwrap();
        assert_eq!(
            decoded.manifest().unwrap().digest.algorithm.as_str(),
            "sha3-256-future"
        );
        let error = decoded.try_into_delaunay().unwrap_err();
        assert!(matches!(
            error,
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::UnsupportedDigestAlgorithm {
                    actual: DelaunayCheckpointDigestAlgorithm::Unknown(ref actual),
                    expected: DelaunayCheckpointDigestAlgorithm::Sha256,
                },
            } if actual == "sha3-256-future"
        ));

        let mut bad_shape = triangle_checkpoint();
        bad_shape["manifest"]["f_vector"] = serde_json::json!([3, 3, 1, 0]);
        let text = serde_json::to_string(&bad_shape).unwrap();
        let mut deserializer = serde_json::Deserializer::from_str(&text);
        let error = DelaunayCheckpoint::<(), (), 2>::decode(&mut deserializer)
            .expect_err("bounded f-vector visitor must reject excess entries");
        assert!(matches!(error, DelaunayCheckpointDecodeError::Codec { .. }));
    }

    #[test]
    fn digest_algorithm_identifiers_are_canonical_across_construction_and_serde() {
        let known = DelaunayCheckpointDigestAlgorithm::from_identifier(
            DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM,
        );
        assert_eq!(known, DelaunayCheckpointDigestAlgorithm::Sha256);
        assert_eq!(
            serde_json::from_str::<DelaunayCheckpointDigestAlgorithm>(
                &serde_json::to_string(&known).unwrap(),
            )
            .unwrap(),
            known
        );

        let future = DelaunayCheckpointDigestAlgorithm::from_identifier("sha3-256-future");
        assert_eq!(future.as_str(), "sha3-256-future");
        assert_eq!(
            serde_json::from_str::<DelaunayCheckpointDigestAlgorithm>(
                &serde_json::to_string(&future).unwrap(),
            )
            .unwrap(),
            future
        );
    }

    #[test]
    fn shape_checked_manifest_retains_the_exact_checked_value() {
        let manifest: DelaunayCheckpointManifest =
            serde_json::from_value(triangle_checkpoint()["manifest"].clone()).unwrap();

        let checked = verify_manifest_shape::<2>(&manifest).unwrap();

        assert!(std::ptr::eq(
            std::ptr::from_ref(checked.manifest()),
            std::ptr::from_ref(&manifest),
        ));
    }

    #[test]
    fn decoded_checkpoint_manifest_accessor_borrows_stored_manifest() {
        let decoded: DelaunayCheckpoint<(), (), 2> =
            serde_json::from_value(triangle_checkpoint()).unwrap();

        let first = decoded.manifest().unwrap();
        let second = decoded.manifest().unwrap();

        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn typed_metadata_error_branches_preserve_categories() {
        let mut missing_manifest = triangle_checkpoint();
        missing_manifest.as_object_mut().unwrap().remove("manifest");
        assert!(matches!(
            typed_checkpoint_error(missing_manifest),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::MissingManifest,
            }
        ));

        let mut manifest_version = triangle_checkpoint();
        manifest_version["manifest"]["manifest_version"] = serde_json::json!(7);
        assert!(matches!(
            typed_checkpoint_error(manifest_version),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::UnsupportedManifestVersion { actual: 7, .. },
            }
        ));

        let mut digest_version = triangle_checkpoint();
        digest_version["manifest"]["digest"]["version"] = serde_json::json!(9);
        assert!(matches!(
            typed_checkpoint_error(digest_version),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::UnsupportedDigestVersion { actual: 9, .. },
            }
        ));

        let mut malformed_digest = triangle_checkpoint();
        malformed_digest["manifest"]["digest"]["value"] = serde_json::json!("0".repeat(63));
        assert!(matches!(
            typed_checkpoint_error(malformed_digest),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::MalformedDigest { .. },
            }
        ));
    }

    #[test]
    fn payload_serializer_custom_message_is_preserved_exactly() {
        let vertices = [
            vertex!([0.0, 0.0]; data = FailingSerializePayload).unwrap(),
            vertex!([1.0, 0.0]; data = FailingSerializePayload).unwrap(),
            vertex!([0.0, 1.0]; data = FailingSerializePayload).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<
            RobustKernel<f64>,
            FailingSerializePayload,
            (),
            2,
        > = DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel(&RobustKernel::new())
            .unwrap();
        assert!(matches!(
            triangulation.checkpoint_manifest(),
            Err(DelaunayCheckpointError::PayloadSerialization { ref message })
                if message == "payload sentinel"
        ));
    }

    #[test]
    fn checkpoint_captures_each_stateful_payload_once() {
        STATEFUL_PAYLOAD_SERIALIZATIONS.with(|serializations| serializations.set(0));
        let vertices = [
            vertex!([0.0, 0.0]; data = StatefulSerializePayload).unwrap(),
            vertex!([1.0, 0.0]; data = StatefulSerializePayload).unwrap(),
            vertex!([0.0, 1.0]; data = StatefulSerializePayload).unwrap(),
        ];
        let mut triangulation: DelaunayTriangulation<
            RobustKernel<f64>,
            StatefulSerializePayload,
            StatefulSerializePayload,
            2,
        > = DelaunayTriangulationBuilder::new(&vertices)
            .simplex_data_type::<StatefulSerializePayload>()
            .build_with_kernel(&RobustKernel::new())
            .unwrap();
        let simplex_key = triangulation.simplices().next().unwrap().0;
        triangulation
            .tri
            .tds
            .set_simplex_data(simplex_key, Some(StatefulSerializePayload))
            .unwrap();

        let checkpoint = serde_json::to_string(&triangulation).unwrap();
        STATEFUL_PAYLOAD_SERIALIZATIONS.with(|serializations| {
            assert_eq!(serializations.get(), (vertices.len() + 1) as u64);
        });

        let restored: DelaunayTriangulation<
            RobustKernel<f64>,
            StatefulSerializePayload,
            StatefulSerializePayload,
            2,
        > = serde_json::from_str(&checkpoint).unwrap();
        restored.validate().unwrap();
        assert!(
            restored
                .vertices()
                .all(|(_, vertex)| vertex.data().is_some())
        );
        assert!(
            restored
                .simplices()
                .all(|(_, simplex)| simplex.data().is_some())
        );
        STATEFUL_PAYLOAD_SERIALIZATIONS.with(|serializations| {
            assert_eq!(serializations.get(), (vertices.len() + 1) as u64);
        });
    }

    #[test]
    fn canonical_payload_range_and_tag_failures_are_typed() {
        assert_eq!(
            canonical_fragment(&i128::MAX).unwrap_err(),
            DelaunayCheckpointError::SignedPayloadIntegerOutOfRange { value: i128::MAX }
        );
        assert_eq!(
            canonical_fragment(&u128::MAX).unwrap_err(),
            DelaunayCheckpointError::UnsignedPayloadIntegerOutOfRange { value: u128::MAX }
        );
        assert_eq!(
            canonical_fragment(&InvalidCborTagPayload).unwrap_err(),
            DelaunayCheckpointError::InvalidPayloadCborTag
        );
    }

    #[test]
    fn checkpoint_decode_errors_retain_conditional_clone_and_equality() {
        #[derive(Clone, Debug, Eq, Error, PartialEq)]
        #[error("codec sentinel")]
        struct CodecSentinel;

        let error = DelaunayCheckpointDecodeError::Codec {
            source: CodecSentinel,
        };
        assert_eq!(error.clone(), error);
    }

    #[test]
    fn json_and_cbor_outer_codecs_restore_the_same_manifest() {
        let triangulation = periodic_checkpoint_fixture();
        let expected = triangulation.checkpoint_manifest().unwrap();
        let json = serde_json::to_vec(&triangulation).unwrap();
        let mut cbor = Vec::new();
        ciborium::ser::into_writer(&triangulation, &mut cbor).unwrap();

        let from_json: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            serde_json::from_slice(&json).unwrap();
        let from_cbor: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            ciborium::de::from_reader(cbor.as_slice()).unwrap();
        assert_eq!(from_json.checkpoint_manifest().unwrap(), expected);
        assert_eq!(from_cbor.checkpoint_manifest().unwrap(), expected);
        from_json.validate().unwrap();
        from_cbor.validate().unwrap();
    }

    #[test]
    fn schema_v1_loader_is_an_executable_migration_bridge() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let original: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let topology_guarantee = original.topology_guarantee().into();
        let validation_policy = original.validation_policy().into();
        let tds = original.into_triangulation().into_tds();
        let legacy: DelaunayTriangulationV1Wire<_, 2> = DelaunayTriangulationV1Wire {
            schema_version: 1,
            tds,
            topology_guarantee,
            global_topology: LegacyGlobalTopologyWire::Euclidean {},
            validation_policy,
        };
        let json_v1 = serde_json::to_vec(&legacy).unwrap();
        let checkpoint: DelaunayCheckpointV1<(), (), 2> = serde_json::from_slice(&json_v1).unwrap();
        let migrated = checkpoint.try_into_delaunay().unwrap();
        migrated.validate().unwrap();

        let json_v2 = serde_json::to_value(&migrated).unwrap();
        assert_eq!(
            json_v2["schema_version"],
            DELAUNAY_CHECKPOINT_SCHEMA_VERSION
        );
        let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            serde_json::from_value(json_v2).unwrap();
        restored.validate().unwrap();
    }

    #[test]
    fn raw_decode_and_v1_envelope_accept_owned_non_datatype_payloads() {
        let marker = OwnedSerdePayload("owned payload".to_owned());
        assert_eq!(marker.0, "owned payload");

        let tds: Tds<OwnedSerdePayload, OwnedSerdePayload, 2> = Tds::empty();
        let encoded = encode_tds_for_test(&tds).unwrap();
        let decoded: Tds<OwnedSerdePayload, OwnedSerdePayload, 2> = decode_tds(&encoded).unwrap();
        assert_eq!(decoded.number_of_vertices(), 0);

        let provenance = replay_checkpoint_provenance(
            &decoded,
            &RobustKernel::new(),
            TopologyGuarantee::Pseudomanifold,
            GlobalTopology::Euclidean,
        )
        .unwrap();
        assert_eq!(provenance, TopologyConstructionProvenance::Unproven);

        let legacy: DelaunayTriangulationV1Wire<_, 2> = DelaunayTriangulationV1Wire {
            schema_version: 1,
            tds: decoded,
            topology_guarantee: TopologyGuaranteeWire::Pseudomanifold,
            global_topology: LegacyGlobalTopologyWire::Euclidean {},
            validation_policy: ValidationPolicyWire::ExplicitOnly,
        };
        let json = serde_json::to_vec(&legacy).unwrap();
        let checkpoint: DelaunayCheckpointV1<OwnedSerdePayload, OwnedSerdePayload, 2> =
            serde_json::from_slice(&json).unwrap();
        assert_eq!(checkpoint.wire.tds.number_of_vertices(), 0);
    }

    #[test]
    fn lower_layer_restoration_accepts_non_exact_kernel() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let original: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let topology_guarantee = original.topology_guarantee();
        let global_topology = original.global_topology();
        let tds = original.into_triangulation().into_tds();

        let (restored, _) = restore_tds_with_provenance(
            tds,
            NonExactKernel(RobustKernel::new()),
            topology_guarantee,
            global_topology,
            TopologyConstructionProvenance::Unproven,
        )
        .unwrap();

        restored.validate().unwrap();
    }

    #[test]
    fn schema_v1_loader_bounds_legacy_toroidal_periods_before_restoration() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let original: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let legacy: DelaunayTriangulationV1Wire<_, 2> = DelaunayTriangulationV1Wire {
            schema_version: 1,
            topology_guarantee: original.topology_guarantee().into(),
            global_topology: LegacyGlobalTopologyWire::Euclidean {},
            validation_policy: original.validation_policy().into(),
            tds: original.into_triangulation().into_tds(),
        };
        let mut value = serde_json::to_value(legacy).unwrap();
        value["global_topology"] = serde_json::json!({
            "kind": "toroidal",
            "periods": [1.0, 1.0, 1.0],
            "mode": "explicit"
        });

        let error = serde_json::from_value::<DelaunayCheckpointV1<(), (), 2>>(value)
            .expect_err("legacy periods must be bounded at the codec boundary");
        assert!(error.to_string().contains("exactly 2 entries"));
    }

    #[test]
    fn global_topology_wire_accepts_unit_variants_without_payload_fields() {
        let euclidean: GlobalTopologyWire<2> =
            serde_json::from_value(serde_json::json!({ "kind": "euclidean" })).unwrap();
        let spherical: GlobalTopologyWire<2> =
            serde_json::from_value(serde_json::json!({ "kind": "spherical" })).unwrap();
        let hyperbolic: GlobalTopologyWire<2> =
            serde_json::from_value(serde_json::json!({ "kind": "hyperbolic" })).unwrap();

        assert!(matches!(euclidean, GlobalTopologyWire::Euclidean {}));
        assert!(matches!(spherical, GlobalTopologyWire::Spherical {}));
        assert!(matches!(hyperbolic, GlobalTopologyWire::Hyperbolic {}));
    }

    #[test]
    fn global_topology_wire_rejects_unknown_fields_for_unit_variants() {
        for kind in ["euclidean", "spherical", "hyperbolic"] {
            let error = serde_json::from_value::<GlobalTopologyWire<2>>(serde_json::json!({
                "kind": kind,
                "unexpected": true
            }))
            .expect_err("unit topology variants must reject payload fields");

            assert!(
                error.to_string().contains("unknown field `unexpected`"),
                "unexpected serde error for {kind}: {error}"
            );
        }
    }

    #[test]
    fn global_topology_wire_rejects_unknown_toroidal_fields() {
        let error = serde_json::from_value::<GlobalTopologyWire<2>>(serde_json::json!({
            "kind": "toroidal",
            "periods": [4_607_182_418_800_017_408_u64, 4_607_182_418_800_017_408_u64],
            "mode": "explicit",
            "unexpected": true
        }))
        .expect_err("toroidal topology must reject fields outside its wire schema");

        assert!(error.to_string().contains("unknown field `unexpected`"));
    }

    #[test]
    fn robust_deserialize_rejects_non_delaunay_connectivity() {
        init_tracing();
        let tds = non_delaunay_quad_tds();
        let topology_guarantee = TopologyGuaranteeWire::PlManifold;
        let global_topology = GlobalTopologyWire::Euclidean {};
        let validation_policy = ValidationPolicyWire::ExplicitOnly;
        let manifest = build_manifest(
            &tds,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
        .unwrap();
        let json = serde_json::to_string(&DelaunayTriangulationSerializeWire {
            schema_version: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
            manifest: manifest.into(),
            tds: encode_tds_for_test(&tds).unwrap(),
            topology_guarantee,
            global_topology,
            validation_policy,
        })
        .unwrap();

        let err =
            serde_json::from_str::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(&json)
                .expect_err("serde reconstruction must reject non-Delaunay connectivity");

        let message = err.to_string();
        assert!(
            message.contains("Delaunay verification failed"),
            "serde error should preserve the Level 5 validation failure: {message}"
        );
    }

    #[test]
    fn robust_deserialize_rejects_legacy_tds_only_json() {
        let json = serde_json::to_string(&Tds::<(), (), 2>::empty()).unwrap();

        let error =
            serde_json::from_str::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(&json)
                .expect_err(
                    "owner-less legacy JSON must not silently acquire default proof context",
                );

        assert!(error.to_string().contains("schema_version"));
    }

    #[test]
    fn robust_deserialize_rejects_unknown_schema_version() {
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        let mut checkpoint = serde_json::to_value(dt).unwrap();
        checkpoint["schema_version"] = serde_json::json!(3);

        let error = serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
            checkpoint,
        )
        .expect_err("unknown checkpoint versions must fail explicitly");

        assert!(
            error
                .to_string()
                .contains("unsupported Delaunay triangulation serialization schema version 3")
        );
    }

    #[test]
    fn robust_deserialize_explicitly_rejects_schema_v1_without_manifest() {
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        let mut checkpoint = serde_json::to_value(dt).unwrap();
        checkpoint["schema_version"] = serde_json::json!(1);
        checkpoint.as_object_mut().unwrap().remove("manifest");

        let error = serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
            checkpoint,
        )
        .expect_err("schema v1 must be rejected rather than migrated without verification");

        assert!(
            error
                .to_string()
                .contains("unsupported Delaunay triangulation serialization schema version 1")
        );
    }

    #[test]
    fn robust_deserialize_parses_toroidal_metadata_before_proof_restoration() {
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        let topology_guarantee = dt.topology_guarantee().into();
        let global_topology = GlobalTopologyWire::Toroidal {
            periods: ExactSlots(vec![1.0_f64.to_bits()]),
            mode: ToroidalConstructionModeWire::Explicit,
        };
        let validation_policy = dt.validation_policy().into();
        let manifest = build_manifest(
            &dt.tri.tds,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
        .unwrap();
        let checkpoint = serde_json::to_value(DelaunayTriangulationSerializeWire {
            schema_version: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
            manifest: manifest.into(),
            tds: encode_tds_for_test(&dt.tri.tds).unwrap(),
            topology_guarantee,
            global_topology,
            validation_policy,
        })
        .unwrap();

        let error = serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
            checkpoint,
        )
        .expect_err("dimension-mismatched raw toroidal metadata must be rejected");

        assert!(error.to_string().contains("exactly 2 entries"));
    }

    #[test]
    fn serialize_internal_delaunay_fixture_does_not_require_kernel_or_datatype_bounds() {
        // No public constructor can express this fixture: the test intentionally
        // uses non-`Kernel`, non-`DataType` parameters to prove the Serialize impl
        // only depends on the serialized TDS payload bounds.
        let dt: DelaunayTriangulation<NotAKernel, SerializeOnlyPayload, SerializeOnlyPayload, 2> =
            DelaunayTriangulation {
                tri: Triangulation {
                    kernel: NotAKernel,
                    tds: Tds::empty(),
                    global_topology: GlobalTopology::DEFAULT,
                    validation_policy: ValidationPolicy::default(),
                    topology_guarantee: TopologyGuarantee::DEFAULT,
                    topology_construction_provenance: TopologyConstructionProvenance::Unproven,
                },
                insertion_state: DelaunayInsertionState::new(),
                spatial_index: None,
                euclidean_report_domain: EuclideanDelaunayReportDomain::Unproven,
            };

        let json = serde_json::to_string(&dt).unwrap();

        assert!(!json.is_empty());
    }

    #[test]
    fn serde_roundtrip_uses_custom_deserialize_impl() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::builder(&vertices)
            .topology_guarantee(TopologyGuarantee::Pseudomanifold)
            .validation_policy(ValidationPolicy::Never)
            .build()
            .unwrap();

        let json = serde_json::to_string(&dt).unwrap();
        let checkpoint: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(checkpoint["schema_version"], 2);
        assert_eq!(checkpoint["manifest"]["manifest_version"], 1);
        assert_eq!(checkpoint["manifest"]["dimension"], 3);
        assert_eq!(
            checkpoint["manifest"]["f_vector"],
            serde_json::json!([4, 6, 4, 1])
        );
        assert_eq!(checkpoint["manifest"]["euler_characteristic"], 1);
        assert_eq!(checkpoint["manifest"]["digest"]["version"], 1);
        assert_eq!(checkpoint["manifest"]["digest"]["algorithm"], "sha256");
        assert_eq!(
            checkpoint["manifest"]["digest"]["value"]
                .as_str()
                .unwrap()
                .len(),
            SHA256_HEX_LENGTH
        );
        assert_eq!(checkpoint["topology_guarantee"], "pseudomanifold");
        assert_eq!(checkpoint["global_topology"]["kind"], "euclidean");
        assert_eq!(checkpoint["validation_policy"], "never");

        let roundtrip_robust: DelaunayTriangulation<RobustKernel<f64>, (), (), 3> =
            serde_json::from_str(&json).unwrap();

        assert_eq!(
            roundtrip_robust.number_of_vertices(),
            dt.number_of_vertices()
        );
        assert_eq!(
            roundtrip_robust.number_of_simplices(),
            dt.number_of_simplices()
        );
        assert_eq!(
            roundtrip_robust.topology_guarantee(),
            TopologyGuarantee::Pseudomanifold
        );
        assert_eq!(
            roundtrip_robust.global_topology(),
            GlobalTopology::Euclidean
        );
        assert_eq!(
            roundtrip_robust.validation_policy(),
            ValidationPolicy::Never
        );
        assert_eq!(
            roundtrip_robust.euclidean_report_domain,
            EuclideanDelaunayReportDomain::Unproven
        );
        assert!(
            roundtrip_robust
                .insertion_state
                .last_inserted_simplex
                .is_none()
        );
    }

    #[test]
    fn manifest_dimension_tampering_is_rejected_before_digest_verification() {
        let mut checkpoint = triangle_checkpoint();
        checkpoint["manifest"]["dimension"] = serde_json::json!(3);

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DimensionMismatch {
                    expected: 2,
                    actual: 3,
                },
            }
        ));
    }

    #[test]
    fn manifest_self_inconsistent_euler_tampering_is_rejected() {
        let mut checkpoint = triangle_checkpoint();
        checkpoint["manifest"]["euler_characteristic"] = serde_json::json!(2);

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::ManifestEulerMismatch {
                    declared: 2,
                    derived: 1,
                },
            }
        ));
    }

    #[test]
    fn every_manifest_f_vector_entry_is_recomputed_from_level_three() {
        let baseline = triangle_checkpoint();
        for index in 0..3 {
            let mut checkpoint = baseline.clone();
            let mut expected_declared = vec![3_u64, 3, 1];
            expected_declared[index] += 1;
            let derived_euler = if index % 2 == 0 { 2 } else { 0 };
            checkpoint["manifest"]["f_vector"] = serde_json::json!(expected_declared);
            checkpoint["manifest"]["euler_characteristic"] = serde_json::json!(derived_euler);

            let DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::FVectorMismatch { declared, computed },
            } = typed_checkpoint_error(checkpoint)
            else {
                panic!("expected a typed f-vector mismatch at index {index}");
            };
            assert_eq!(declared, expected_declared, "f-vector index {index}");
            assert_eq!(computed, [3, 3, 1], "f-vector index {index}");
        }
    }

    #[test]
    fn manifest_digest_tampering_is_rejected() {
        let mut checkpoint = triangle_checkpoint();
        checkpoint["manifest"]["digest"]["value"] = serde_json::json!("0".repeat(64));

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    #[test]
    fn embedded_tds_image_rejects_trailing_bytes() {
        let mut checkpoint = triangle_checkpoint();
        checkpoint["tds"]
            .as_array_mut()
            .expect("JSON checkpoint should encode TDS bytes as an array")
            .push(serde_json::json!(0xf6));

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::TrailingTdsBytes { count: 1 },
            }
        ));
    }

    #[test]
    fn embedded_tds_image_preserves_codec_and_hydration_categories() {
        let mut invalid_cbor = triangle_checkpoint();
        invalid_cbor["tds"] = serde_json::json!([0xff]);
        assert!(matches!(
            typed_checkpoint_error(invalid_cbor),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::TdsCodec { ref message },
            } if !message.is_empty()
        ));

        let mut invalid_snapshot = triangle_checkpoint();
        mutate_embedded_tds(&mut invalid_snapshot, |tds| {
            cbor_field_mut(tds, "simplex_vertices")
                .as_map_mut()
                .expect("simplex vertex relationships should be a map")
                .clear();
        });
        let DelaunayCheckpointLoadError::Checkpoint {
            source: DelaunayCheckpointError::TdsHydration { source },
        } = typed_checkpoint_error(invalid_snapshot)
        else {
            panic!("expected a typed missing simplex-vertex relationship");
        };
        let DelaunayCheckpointTdsHydrationError::MissingSimplexVertexUuids { simplex_uuid } =
            *source
        else {
            panic!("expected a typed missing simplex-vertex relationship");
        };
        assert!(!simplex_uuid.is_nil());
    }

    #[test]
    fn canonical_digest_rejects_coordinate_and_proof_context_tampering() {
        let mut coordinate_checkpoint = triangle_checkpoint();
        mutate_embedded_tds(&mut coordinate_checkpoint, |tds| {
            let vertex = &mut cbor_field_mut(tds, "vertices").as_array_mut().unwrap()[0];
            cbor_field_mut(vertex, "point").as_array_mut().unwrap()[0] = Value::Float(0.125);
        });
        assert!(matches!(
            typed_checkpoint_error(coordinate_checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));

        let mut context_checkpoint = triangle_checkpoint();
        context_checkpoint["validation_policy"] = serde_json::json!("always");
        assert!(matches!(
            typed_checkpoint_error(context_checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    #[test]
    fn canonical_digest_rejects_uuid_connectivity_tampering() {
        let mut checkpoint = triangle_checkpoint();
        mutate_embedded_tds(&mut checkpoint, |tds| {
            let vertex_slots = cbor_field_mut(tds, "simplex_vertices")
                .as_map_mut()
                .unwrap()
                .first_mut()
                .unwrap()
                .1
                .as_array_mut()
                .unwrap();
            vertex_slots.swap(0, 1);
        });

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    #[test]
    fn canonical_digest_includes_user_payload_fields() {
        let vertices = [
            vertex!([0.0, 0.0]; data = 10_u32).unwrap(),
            vertex!([1.0, 0.0]; data = 20_u32).unwrap(),
            vertex!([0.0, 1.0]; data = 30_u32).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, u32, (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let mut checkpoint = serde_json::to_value(triangulation).unwrap();
        mutate_embedded_tds(&mut checkpoint, |tds| {
            let vertex = &mut cbor_field_mut(tds, "vertices").as_array_mut().unwrap()[0];
            *cbor_field_mut(vertex, "data") = Value::Integer(99_i32.into());
        });

        assert!(matches!(
            typed_checkpoint_error_with_payload::<u32, ()>(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    #[test]
    fn distinct_owner_state_with_the_same_f_vector_fails_digest_verification() {
        let mut first = triangle_checkpoint();
        let second_vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(),
            vertex!([0.0, 2.0]).unwrap(),
        ];
        let second: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::builder(&second_vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let second = serde_json::to_value(second).unwrap();

        assert_eq!(
            first["manifest"]["f_vector"],
            second["manifest"]["f_vector"]
        );
        first["tds"] = second["tds"].clone();

        assert!(matches!(
            typed_checkpoint_error(first),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    #[test]
    fn canonical_digest_is_independent_of_snapshot_record_order() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(),
            vertex!([2.0, 1.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::builder(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let mut checkpoint = serde_json::to_value(triangulation).unwrap();
        mutate_embedded_tds(&mut checkpoint, |tds| {
            cbor_field_mut(tds, "vertices")
                .as_array_mut()
                .unwrap()
                .reverse();
            cbor_field_mut(tds, "simplices")
                .as_array_mut()
                .unwrap()
                .reverse();
        });

        let restored =
            serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
                checkpoint,
            )
            .expect("UUID-sorted digest must ignore snapshot record order");
        restored.validate().unwrap();
    }

    #[test]
    fn checkpoint_digest_survives_json_float_roundtrip() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([0.0, -22.546_422_723_221_383]).unwrap(),
            vertex!([12.252_739_760_228_783, 0.0]).unwrap(),
            vertex!([1.572_332_120_964_092_1, 2.883_274_933_268_964]).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::builder(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let checkpoint = serde_json::to_string(&triangulation).unwrap();

        let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            serde_json::from_str(&checkpoint).unwrap();
        restored.validate().unwrap();
    }

    #[test]
    fn canonical_user_payload_maps_are_independent_of_serialization_order() {
        let forward = streaming_payload_bytes(&MapOrderPayload { reverse: false }).unwrap();
        let reverse = streaming_payload_bytes(&MapOrderPayload { reverse: true }).unwrap();

        assert_eq!(forward, reverse);
    }

    #[test]
    fn streaming_payload_encoder_preserves_digest_v1_serde_shapes() {
        assert_streaming_payload_compatibility(&CompositeSerdePayload {
            signed: -37,
            unsigned: 41,
            float: 1.25,
            character: 'λ',
            sequence: [2, 3, 5],
            optional: Some(8),
        });
        assert_streaming_payload_compatibility(&SerdeShapePayload::Unit);
        assert_streaming_payload_compatibility(&SerdeShapePayload::Newtype(13));
        assert_streaming_payload_compatibility(&SerdeShapePayload::Tuple(17, true));
        assert_streaming_payload_compatibility(&SerdeShapePayload::Struct {
            count: 19,
            enabled: false,
        });
        assert_streaming_payload_compatibility(&ciborium::tag::Required::<u32, 42>(23));
        assert_streaming_payload_compatibility(&Value::Bytes(vec![29, 31]));

        let map = BTreeMap::from([("zeta", 37_u32), ("alpha", 41_u32)]);
        assert_streaming_payload_compatibility(&map);
    }

    #[test]
    fn duplicate_canonical_payload_map_keys_are_rejected_in_every_entry_order() {
        for reverse in [false, true] {
            assert_eq!(
                streaming_payload_bytes(&DuplicateKeyPayload { reverse })
                    .expect_err("last-key-wins payload maps must not enter digest-v1"),
                DelaunayCheckpointError::DuplicatePayloadMapKey
            );
        }

        let vertices = [
            vertex!([0.0, 0.0]; data = DuplicateKeyPayload { reverse: false }).unwrap(),
            vertex!([1.0, 0.0]; data = DuplicateKeyPayload { reverse: false }).unwrap(),
            vertex!([0.0, 1.0]; data = DuplicateKeyPayload { reverse: false }).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, DuplicateKeyPayload, (), 2> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        assert_eq!(
            triangulation
                .checkpoint_manifest()
                .expect_err("checkpoint construction must reject duplicate payload keys"),
            DelaunayCheckpointError::DuplicatePayloadMapKey
        );
    }

    #[test]
    fn digest_v1_fixed_triangle_vector_is_stable() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let coordinates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut vertex_keys = Vec::new();
        for (index, coordinates) in coordinates.into_iter().enumerate() {
            let point = *vertex!(coordinates).unwrap().point();
            let uuid = Uuid::parse_str(&format!(
                "00000000-0000-4000-8000-{value:012}",
                value = index + 1
            ))
            .unwrap();
            let vertex = Vertex::from_validated_point_with_uuid(point, uuid, None);
            vertex_keys.push(tds.insert_vertex_with_mapping(vertex).unwrap());
        }
        let simplex_uuid = Uuid::parse_str("00000000-0000-4000-8000-000000000010").unwrap();
        let simplex = Simplex::try_new_with_uuid(vertex_keys, simplex_uuid, None).unwrap();
        tds.insert_simplex_with_mapping(simplex).unwrap();
        tds.force_construction_complete_for_test();
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        let manifest = build_manifest(
            &tds,
            TopologyGuaranteeWire::PlManifold,
            &GlobalTopologyWire::Euclidean {},
            ValidationPolicyWire::ExplicitOnly,
        )
        .unwrap();
        assert_eq!(
            manifest.digest.value,
            "861784a500ae8683fd3439abb842d4676c6f976f8fb792026edea82402f72708"
        );
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    fn representative_manifest_allocation_budget_is_bounded() {
        let vertices = (0_u32..64)
            .map(|index| {
                let row = index / 8;
                let column = index % 8;
                let jitter = f64::from((row * 17 + column * 31) % 11) * 1.0e-4;
                vertex!([f64::from(column) + jitter, f64::from(row) - jitter]).unwrap()
            })
            .collect::<Vec<_>>();
        let triangulation = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .unwrap();
        let ((), allocations) = measure_with_result(|| {
            drop(triangulation.checkpoint_manifest().unwrap());
        });
        assert!(
            allocations.count_total < 2_000,
            "representative manifest exceeded allocation budget: {allocations:?}"
        );
        assert_eq!(allocations.count_current, 0, "{allocations:?}");
        assert_eq!(allocations.bytes_current, 0, "{allocations:?}");
    }

    #[cfg(feature = "count-allocations")]
    #[test]
    fn decoded_checkpoint_manifest_inspection_is_zero_allocation() {
        let decoded: DelaunayCheckpoint<(), (), 2> =
            serde_json::from_value(triangle_checkpoint()).unwrap();

        let (dimension, allocations) =
            measure_with_result(|| decoded.manifest().unwrap().dimension);

        assert_eq!(dimension, 2);
        assert_eq!(allocations.count_total, 0, "{allocations:?}");
        assert_eq!(allocations.bytes_total, 0, "{allocations:?}");
        assert_eq!(allocations.count_current, 0, "{allocations:?}");
        assert_eq!(allocations.bytes_current, 0, "{allocations:?}");
    }

    #[test]
    fn periodic_checkpoint_roundtrip_and_offset_tampering_are_verified() {
        let triangulation = periodic_checkpoint_fixture();
        let manifest = triangulation.checkpoint_manifest().unwrap();
        assert_eq!(manifest.dimension, 2);
        assert_eq!(manifest.euler_characteristic, 0);
        assert_eq!(manifest.f_vector, [7, 21, 14]);

        let mut checkpoint = serde_json::to_value(triangulation).unwrap();
        let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            serde_json::from_value(checkpoint.clone()).unwrap();
        assert!(restored.global_topology().is_toroidal());
        assert_eq!(restored.checkpoint_manifest().unwrap(), manifest);
        restored.validate().unwrap();

        mutate_embedded_tds(&mut checkpoint, |tds| {
            let offsets = cbor_field_mut(tds, "simplex_vertex_offsets")
                .as_map_mut()
                .unwrap()
                .first_mut()
                .unwrap()
                .1
                .as_array_mut()
                .unwrap();
            for offset in offsets {
                for component in offset.as_array_mut().unwrap() {
                    let shifted = i64::try_from(component.as_integer().unwrap()).unwrap() + 1;
                    *component = Value::Integer(shifted.into());
                }
            }
        });

        assert!(matches!(
            typed_checkpoint_error(checkpoint),
            DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            }
        ));
    }

    macro_rules! assert_single_simplex_manifest {
        ($dim:literal, [$([$($coordinate:expr),+ $(,)?]),+ $(,)?], [$($count:expr),+ $(,)?]) => {{
            let vertices = [$(vertex!([$($coordinate),+]).unwrap()),+];
            let triangulation = DelaunayTriangulation::builder(&vertices).build().unwrap();
            let manifest = triangulation.checkpoint_manifest().unwrap();
            assert_eq!(manifest.dimension, $dim);
            assert_eq!(manifest.f_vector, vec![$($count),+]);
            assert_eq!(manifest.euler_characteristic, 1);
            assert_eq!(manifest.digest.value.len(), SHA256_HEX_LENGTH);
        }};
    }

    #[test]
    fn ordinary_checkpoint_manifests_cover_dimensions_one_through_five() {
        assert_single_simplex_manifest!(1, [[0.0], [1.0]], [2, 1]);
        assert_single_simplex_manifest!(2, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [3, 3, 1]);
        assert_single_simplex_manifest!(
            3,
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [4, 6, 4, 1]
        );
        assert_single_simplex_manifest!(
            4,
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [5, 10, 10, 5, 1]
        );
        assert_single_simplex_manifest!(
            5,
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [6, 15, 20, 15, 6, 1]
        );
    }

    #[test]
    fn provenance_signature_mismatches_preserve_the_differing_category_and_evidence() {
        let first = Uuid::from_u128(1);
        let second = Uuid::from_u128(2);
        let third = Uuid::from_u128(3);
        let checkpoint_vertices = [(first, [10_u64, 20_u64])];
        let replayed_vertices = [(first, [10_u64, 21_u64])];

        let vertex_error =
            verify_provenance_signatures(&checkpoint_vertices, &replayed_vertices, &[], &[])
                .unwrap_err();
        assert_eq!(
            vertex_error,
            DelaunayCheckpointLoadError::ProvenanceVertexMismatch {
                index: 0,
                checkpoint_vertex_uuid: Some(first),
                checkpoint_coordinate_bits: Some(vec![10, 20]),
                replayed_vertex_uuid: Some(first),
                replayed_coordinate_bits: Some(vec![10, 21]),
                checkpoint_count: 1,
                replayed_count: 1,
            }
        );
        assert_eq!(vertex_error.clone(), vertex_error);

        let shared_vertices = [(first, [10_u64, 20_u64])];
        let checkpoint_simplices = [vec![first, second]];
        let replayed_simplices = [vec![first, third]];
        assert_eq!(
            verify_provenance_signatures(
                &shared_vertices,
                &shared_vertices,
                &checkpoint_simplices,
                &replayed_simplices,
            )
            .unwrap_err(),
            DelaunayCheckpointLoadError::ProvenanceSimplexMismatch {
                index: 0,
                checkpoint_vertices: Some(vec![first, second]),
                replayed_vertices: Some(vec![first, third]),
                checkpoint_count: 1,
                replayed_count: 1,
            }
        );
    }

    #[test]
    fn multi_simplex_four_and_five_dimensional_checkpoints_replay_provenance() {
        let vertices_4d = [
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.1, 0.1, 0.1, 0.1]).unwrap(),
        ];
        let triangulation_4d: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
            DelaunayTriangulationBuilder::new(&vertices_4d)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        assert!(triangulation_4d.number_of_simplices() > 1);
        let manifest_4d = triangulation_4d.checkpoint_manifest().unwrap();
        let vertices_4d = vertex_signatures(&triangulation_4d.tri.tds);
        let simplices_4d = euclidean_simplex_signatures(&triangulation_4d.tri.tds).unwrap();
        let json_4d = serde_json::to_string(&triangulation_4d).unwrap();
        let restored_4d: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
            serde_json::from_str(&json_4d).unwrap();
        assert_eq!(restored_4d.checkpoint_manifest().unwrap(), manifest_4d);
        assert_eq!(vertex_signatures(&restored_4d.tri.tds), vertices_4d);
        assert_eq!(
            euclidean_simplex_signatures(&restored_4d.tri.tds).unwrap(),
            simplices_4d
        );
        restored_4d.validate().unwrap();

        let vertices_5d = [
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.1, 0.1, 0.1, 0.1, 0.1]).unwrap(),
        ];
        let triangulation_5d: DelaunayTriangulation<RobustKernel<f64>, (), (), 5> =
            DelaunayTriangulationBuilder::new(&vertices_5d)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        assert!(triangulation_5d.number_of_simplices() > 1);
        let manifest_5d = triangulation_5d.checkpoint_manifest().unwrap();
        let vertices_5d = vertex_signatures(&triangulation_5d.tri.tds);
        let simplices_5d = euclidean_simplex_signatures(&triangulation_5d.tri.tds).unwrap();
        let mut checkpoint_5d = serde_json::to_value(&triangulation_5d).unwrap();
        let restored_5d: DelaunayTriangulation<RobustKernel<f64>, (), (), 5> =
            serde_json::from_value(checkpoint_5d.clone()).unwrap();
        assert_eq!(restored_5d.checkpoint_manifest().unwrap(), manifest_5d);
        assert_eq!(vertex_signatures(&restored_5d.tri.tds), vertices_5d);
        assert_eq!(
            euclidean_simplex_signatures(&restored_5d.tri.tds).unwrap(),
            simplices_5d
        );
        restored_5d.validate().unwrap();

        mutate_embedded_tds(&mut checkpoint_5d, |tds| {
            let vertex = &mut cbor_field_mut(tds, "vertices").as_array_mut().unwrap()[0];
            cbor_field_mut(vertex, "point").as_array_mut().unwrap()[4] = Value::Float(0.125);
        });
        let checkpoint: DelaunayCheckpoint<(), (), 5> =
            serde_json::from_value(checkpoint_5d).unwrap();
        assert!(matches!(
            checkpoint.try_into_delaunay(),
            Err(DelaunayCheckpointLoadError::Checkpoint {
                source: DelaunayCheckpointError::DigestMismatch { .. },
            })
        ));
    }

    #[test]
    fn periodic_four_and_five_dimensional_checkpoints_reject_forged_provenance() {
        let vertices = [
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.1, 0.1, 0.1, 0.1]).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 4> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let topology_guarantee = TopologyGuaranteeWire::PlManifold;
        let global_topology = GlobalTopologyWire::Toroidal {
            periods: ExactSlots(vec![1.0_f64.to_bits(); 4]),
            mode: ToroidalConstructionModeWire::PeriodicImagePoint,
        };
        let validation_policy = ValidationPolicyWire::ExplicitOnly;
        let manifest = build_manifest(
            &triangulation.tri.tds,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
        .unwrap();
        let tds = encode_tds_for_test(&triangulation.tri.tds).unwrap();
        let checkpoint = DelaunayCheckpoint::<(), (), 4> {
            wire: DelaunayTriangulationWire {
                schema_version: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
                manifest: Some(manifest.into()),
                tds,
                topology_guarantee,
                global_topology,
                validation_policy,
            },
            payload: PhantomData,
        };
        assert!(matches!(
            checkpoint.try_into_delaunay(),
            Err(DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance { dimension: 4 })
        ));

        let vertices = [
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            vertex!([0.1, 0.1, 0.1, 0.1, 0.1]).unwrap(),
        ];
        let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 5> =
            DelaunayTriangulationBuilder::new(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let topology_guarantee = TopologyGuaranteeWire::PlManifold;
        let global_topology = GlobalTopologyWire::Toroidal {
            periods: ExactSlots(vec![1.0_f64.to_bits(); 5]),
            mode: ToroidalConstructionModeWire::PeriodicImagePoint,
        };
        let validation_policy = ValidationPolicyWire::ExplicitOnly;
        let manifest = build_manifest(
            &triangulation.tri.tds,
            topology_guarantee,
            &global_topology,
            validation_policy,
        )
        .unwrap();
        let tds = encode_tds_for_test(&triangulation.tri.tds).unwrap();
        let checkpoint = DelaunayCheckpoint::<(), (), 5> {
            wire: DelaunayTriangulationWire {
                schema_version: DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
                manifest: Some(manifest.into()),
                tds,
                topology_guarantee,
                global_topology,
                validation_policy,
            },
            payload: PhantomData,
        };
        assert!(matches!(
            checkpoint.try_into_delaunay(),
            Err(DelaunayCheckpointLoadError::UnsupportedPeriodicProvenance { dimension: 5 })
        ));
    }

    #[test]
    fn ambiguous_option_unit_payloads_are_rejected_for_vertices_and_simplices() {
        for data in [None, Some(())] {
            let vertices = [
                vertex!([0.0, 0.0]; data = data).unwrap(),
                vertex!([1.0, 0.0]; data = data).unwrap(),
                vertex!([0.0, 1.0]; data = data).unwrap(),
            ];
            let triangulation: DelaunayTriangulation<RobustKernel<f64>, Option<()>, (), 2> =
                DelaunayTriangulationBuilder::new(&vertices)
                    .build_with_kernel(&RobustKernel::new())
                    .unwrap();
            let error = serde_json::to_value(triangulation)
                .expect_err("ambiguous vertex payload must be rejected");
            assert!(error.to_string().contains("null or unit state"));
        }

        for data in [None, Some(())] {
            let mut tds: Tds<(), Option<()>, 2> = Tds::empty();
            let v0 = tds
                .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
                .unwrap();
            let v1 = tds
                .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
                .unwrap();
            let v2 = tds
                .insert_vertex_with_mapping(vertex!([0.0, 1.0]).unwrap())
                .unwrap();
            tds.insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], Some(data)).unwrap(),
            )
            .unwrap();
            tds.force_construction_complete_for_test();
            tds.assign_neighbors().unwrap();
            tds.assign_incident_simplices().unwrap();
            let error = build_manifest(
                &tds,
                TopologyGuaranteeWire::PlManifold,
                &GlobalTopologyWire::Euclidean {},
                ValidationPolicyWire::ExplicitOnly,
            )
            .expect_err("ambiguous simplex payload must be rejected");
            assert!(matches!(error, DelaunayCheckpointError::AmbiguousPayload));
        }
    }

    #[test]
    fn valid_bistellar_move_changes_subdivision_evidence_but_preserves_euler() {
        let vertices = [
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let before: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            DelaunayTriangulation::builder(&vertices)
                .build_with_kernel(&RobustKernel::new())
                .unwrap();
        let before_manifest = before.checkpoint_manifest().unwrap();

        let mut moved = before.into_triangulation();
        let simplex_key = moved.simplices().next().unwrap().0;
        moved
            .flip_k1_insert(simplex_key, vertex!([0.25, 0.25]).unwrap())
            .unwrap();
        let after = DelaunayRefinementBuilder::new(moved).build().unwrap();
        let after_manifest = after.checkpoint_manifest().unwrap();

        assert_eq!(before_manifest.f_vector, vec![3, 3, 1]);
        assert_eq!(after_manifest.f_vector, vec![4, 6, 3]);
        assert_ne!(before_manifest.digest, after_manifest.digest);
        assert_eq!(before_manifest.euler_characteristic, 1);
        assert_eq!(after_manifest.euler_characteristic, 1);

        let checkpoint = serde_json::to_string(&after).unwrap();
        let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
            serde_json::from_str(&checkpoint).unwrap();
        assert_eq!(restored.checkpoint_manifest().unwrap(), after_manifest);
    }
}
