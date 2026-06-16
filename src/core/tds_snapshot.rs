//! Durable UUID snapshots for [`Tds`] persistence boundaries.
//!
//! The runtime [`Tds`] stores topology with slotmap keys because those handles
//! are compact and fast in memory. A snapshot stores the same topology with
//! vertex and simplex UUIDs so the data can cross process, file, or codec
//! boundaries without treating storage-local keys as durable identifiers.
//!
//! This module keeps three roles separate:
//!
//! - `RawTdsSnapshot` is the codec-facing interchange record. It may come from
//!   untrusted input and can temporarily contain cross-field inconsistencies.
//! - `TdsSnapshot` is the validated UUID topology. It is proof-bearing: every
//!   simplex has checked vertex UUIDs, neighbor UUID slots, and optional periodic
//!   offsets before hydration starts.
//! - `Tds` is the runtime slotmap-backed topology. Hydration allocates fresh
//!   storage-local keys from a validated snapshot, rebuilds incidence, then runs
//!   TDS validation before returning the value.
//!
//! Downstream crates should normally use `Serialize`/`Deserialize` on `Tds<U, V, D>`
//! instead of these crate-private records. That path preserves vertex payloads
//! (`U`) and simplex payloads (`V`) whenever those types satisfy the crate's data
//! serialization bounds, while keeping `VertexKey` and `SimplexKey` out of the
//! durable format. The raw snapshot shape is serde-backed today, but its role is
//! a persistence boundary rather than a serde-specific domain model.

#![forbid(unsafe_code)]

use super::{
    SimplexKey, Tds, TdsError, TdsMutationError, TriangulationConstructionState, VertexKey,
};
use crate::core::{
    collections::{
        Entry, FastHashMap, FastHashSet, NeighborBuffer, PeriodicOffsetBuffer,
        SimplexVertexUuidBuffer, StorageMap, UuidToSimplexKeyMap, UuidToVertexKeyMap,
        fast_hash_map_with_capacity, fast_hash_set_with_capacity,
    },
    simplex::{Simplex, SimplexValidationError},
    traits::{DataDeserialize, DataSerialize},
    util::validate_uuid,
    vertex::Vertex,
};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{self, MapAccess, Visitor},
    ser::SerializeStruct,
};
use std::{
    fmt,
    marker::PhantomData,
    sync::{Arc, atomic::AtomicU64},
};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// SNAPSHOT ERROR TYPES
// =============================================================================

/// Errors that can occur while building or parsing durable TDS snapshots.
///
/// Snapshots use UUIDs as interchange identities. Parsing a snapshot resolves
/// those UUIDs into fresh storage-local slotmap keys and rejects incomplete or
/// inconsistent topology before constructing a live [`Tds`].
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
enum TdsSnapshotError {
    /// A simplex could not resolve one of its vertex keys to a vertex UUID while
    /// building a snapshot.
    #[error("Could not resolve vertex UUIDs for simplex {simplex_uuid}: {source}")]
    SimplexVertexUuidResolutionFailed {
        /// Simplex whose vertex UUIDs could not be resolved.
        simplex_uuid: Uuid,
        /// Structured simplex validation failure.
        #[source]
        source: SimplexValidationError,
    },
    /// A simplex has no assigned neighbor slots while building a snapshot.
    #[error("No assigned neighbor slots found for simplex {simplex_uuid}")]
    MissingSimplexNeighborSlots {
        /// Simplex whose runtime neighbor slots are absent.
        simplex_uuid: Uuid,
    },
    /// A runtime neighbor key could not be resolved to a simplex UUID.
    #[error(
        "Neighbor key {neighbor_key:?} referenced by simplex {simplex_uuid} not found in simplices"
    )]
    DanglingRuntimeNeighborKey {
        /// Simplex containing the dangling runtime neighbor key.
        simplex_uuid: Uuid,
        /// Neighbor key that could not be resolved.
        neighbor_key: SimplexKey,
    },
    /// The runtime TDS failed validation before snapshot serialization.
    #[error("Source TDS failed validation before snapshot serialization: {source}")]
    SourceValidationFailed {
        /// Structured validation failure from the source TDS.
        #[source]
        source: TdsError,
    },
    /// A snapshot vertex UUID appeared more than once.
    #[error("Duplicate vertex UUID {vertex_uuid} in TDS snapshot vertices")]
    DuplicateVertexUuid {
        /// Duplicate vertex UUID from the snapshot vertex records.
        vertex_uuid: Uuid,
    },
    /// A snapshot simplex record had no matching vertex-UUID relationship.
    #[error("No vertex UUIDs found for simplex {simplex_uuid}")]
    MissingSimplexVertexUuids {
        /// Simplex UUID missing from the `simplex_vertices` relationship map.
        simplex_uuid: Uuid,
    },
    /// A snapshot simplex record had no matching neighbor-UUID relationship.
    #[error("No neighbor UUIDs found for simplex {simplex_uuid}")]
    MissingSimplexNeighborUuids {
        /// Simplex UUID missing from the `simplex_neighbors` relationship map.
        simplex_uuid: Uuid,
    },
    /// A simplex's serialized vertex UUID slots did not contain exactly `D + 1`
    /// entries.
    #[error(
        "Simplex {simplex_uuid} has {actual} vertex UUID slots in snapshot, expected {expected}"
    )]
    InvalidSimplexVertexUuidSlotCount {
        /// Simplex whose vertex UUID slot count is malformed.
        simplex_uuid: Uuid,
        /// Number of vertex UUID slots present in the snapshot.
        actual: usize,
        /// Expected number of vertex UUID slots (`D + 1`).
        expected: usize,
    },
    /// A simplex referenced a vertex UUID that was not present in the snapshot.
    #[error("Vertex UUID {vertex_uuid} referenced by simplex {simplex_uuid} not found in vertices")]
    DanglingSimplexVertexUuid {
        /// Simplex containing the dangling vertex reference.
        simplex_uuid: Uuid,
        /// Vertex UUID that could not be resolved to a snapshot vertex.
        vertex_uuid: Uuid,
    },
    /// A simplex referenced a neighbor UUID that was not present in the snapshot.
    #[error(
        "Neighbor UUID {neighbor_uuid} referenced by simplex {simplex_uuid} not found in simplices"
    )]
    DanglingSimplexNeighborUuid {
        /// Simplex containing the dangling neighbor reference.
        simplex_uuid: Uuid,
        /// Neighbor UUID that could not be resolved to a snapshot simplex.
        neighbor_uuid: Uuid,
    },
    /// A simplex could not be constructed from its resolved vertex keys or
    /// neighbor keys.
    #[error("Invalid snapshot simplex {simplex_uuid}: {source}")]
    InvalidSimplex {
        /// UUID of the simplex being reconstructed.
        simplex_uuid: Uuid,
        /// Structured simplex validation failure.
        #[source]
        source: SimplexValidationError,
    },
    /// A snapshot simplex UUID appeared more than once.
    #[error("Duplicate simplex UUID {simplex_uuid} in TDS snapshot simplices")]
    DuplicateSimplexUuid {
        /// Duplicate simplex UUID from the snapshot simplex records.
        simplex_uuid: Uuid,
    },
    /// The vertex-UUID relationship map mentioned an unknown simplex.
    #[error("Vertex UUID mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexVertexMapping {
        /// Unknown simplex UUID present in `simplex_vertices`.
        simplex_uuid: Uuid,
    },
    /// The neighbor-UUID relationship map mentioned an unknown simplex.
    #[error("Neighbor UUID mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexNeighborMapping {
        /// Unknown simplex UUID present in `simplex_neighbors`.
        simplex_uuid: Uuid,
    },
    /// The periodic-offset relationship map mentioned an unknown simplex.
    #[error("Periodic offset mapping provided for unknown simplex {simplex_uuid}")]
    UnknownSimplexOffsetMapping {
        /// Unknown simplex UUID present in `simplex_vertex_offsets`.
        simplex_uuid: Uuid,
    },
    /// A serialized periodic offset had the wrong coordinate dimension.
    #[error(
        "Periodic offset {offset_index} for simplex {simplex_uuid} has dimension {actual}, expected {expected}"
    )]
    PeriodicOffsetDimensionMismatch {
        /// Simplex whose offset record is malformed.
        simplex_uuid: Uuid,
        /// Offset index within the simplex-local offset list.
        offset_index: usize,
        /// Expected coordinate dimension.
        expected: usize,
        /// Observed coordinate dimension.
        actual: usize,
    },
    /// Rebuilding vertex incident-simplex pointers from snapshot topology failed.
    #[error("Failed to rebuild TDS vertex incidence from snapshot: {source}")]
    IncidentSimplexRebuildFailed {
        /// Structured TDS mutation failure from incident-simplex assignment.
        #[source]
        source: TdsMutationError,
    },
    /// Final TDS validation failed after UUID relationships were resolved.
    #[error("TDS snapshot failed validation: {source}")]
    ValidationFailed {
        /// Structured validation failure from the rebuilt TDS.
        #[source]
        source: TdsError,
    },
}

// =============================================================================
// RAW SNAPSHOT RECORD TYPES
// =============================================================================

/// Raw durable UUID-based image of a TDS topology.
///
/// This type is intentionally heavier than the runtime TDS. It is for I/O and
/// codec boundaries only, where stable UUID relationships are more important
/// than slotmap-key locality. It can contain cross-field inconsistencies until
/// parsed into [`TdsSnapshot`].
#[derive(Debug, Deserialize, Serialize)]
#[serde(
    bound(
        serialize = "U: DataSerialize, V: DataSerialize",
        deserialize = "U: DataDeserialize, V: DataDeserialize"
    ),
    deny_unknown_fields
)]
struct RawTdsSnapshot<U, V, const D: usize> {
    vertices: Vec<Vertex<U, D>>,
    simplices: Vec<RawSnapshotSimplex<V>>,
    #[serde(deserialize_with = "deserialize_simplex_vertices_no_duplicates")]
    simplex_vertices: FastHashMap<Uuid, Vec<Uuid>>,
    #[serde(deserialize_with = "deserialize_simplex_neighbors_no_duplicates")]
    simplex_neighbors: FastHashMap<Uuid, Vec<Option<Uuid>>>,
    #[serde(
        default,
        deserialize_with = "deserialize_simplex_vertex_offsets_no_duplicates"
    )]
    simplex_vertex_offsets: FastHashMap<Uuid, Vec<Vec<i8>>>,
}

fn deserialize_simplex_vertices_no_duplicates<'de, De>(
    deserializer: De,
) -> Result<FastHashMap<Uuid, Vec<Uuid>>, De::Error>
where
    De: Deserializer<'de>,
{
    deserialize_uuid_map_no_duplicates("simplex_vertices", deserializer)
}

fn deserialize_simplex_neighbors_no_duplicates<'de, De>(
    deserializer: De,
) -> Result<FastHashMap<Uuid, Vec<Option<Uuid>>>, De::Error>
where
    De: Deserializer<'de>,
{
    deserialize_uuid_map_no_duplicates("simplex_neighbors", deserializer)
}

fn deserialize_simplex_vertex_offsets_no_duplicates<'de, De>(
    deserializer: De,
) -> Result<FastHashMap<Uuid, Vec<Vec<i8>>>, De::Error>
where
    De: Deserializer<'de>,
{
    deserialize_uuid_map_no_duplicates("simplex_vertex_offsets", deserializer)
}

/// Deserializes a UUID relationship map while rejecting duplicate simplex keys.
///
/// This protects the public [`Tds`] deserialization contract: untrusted snapshot
/// input must not be able to rely on codec-level "last key wins" behavior to
/// replace vertex, neighbor, or periodic-offset relationships silently.
fn deserialize_uuid_map_no_duplicates<'de, De, T>(
    field_name: &'static str,
    deserializer: De,
) -> Result<FastHashMap<Uuid, T>, De::Error>
where
    De: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct UuidMapVisitor<T> {
        field_name: &'static str,
        _phantom: PhantomData<T>,
    }

    impl<'de, T> Visitor<'de> for UuidMapVisitor<T>
    where
        T: Deserialize<'de>,
    {
        type Value = FastHashMap<Uuid, T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(
                formatter,
                "a UUID-keyed TDS snapshot `{}` relationship map",
                self.field_name
            )
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut values = fast_hash_map_with_capacity(map.size_hint().unwrap_or(0));

            while let Some((simplex_uuid, value)) = map.next_entry::<Uuid, T>()? {
                match values.entry(simplex_uuid) {
                    Entry::Occupied(entry) => {
                        return Err(de::Error::custom(format!(
                            "duplicate simplex UUID key `{}` in TDS snapshot `{}` relationship map",
                            entry.key(),
                            self.field_name
                        )));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(value);
                    }
                }
            }

            Ok(values)
        }
    }

    deserializer.deserialize_map(UuidMapVisitor {
        field_name,
        _phantom: PhantomData,
    })
}

/// Raw snapshot simplex record used by [`RawTdsSnapshot`].
///
/// The record stores only durable simplex identity and optional payload. TDS
/// topology relationships live in `RawTdsSnapshot`'s UUID maps.
#[derive(Debug)]
struct RawSnapshotSimplex<V> {
    uuid: Uuid,
    data: Option<V>,
}

// =============================================================================
// VALIDATED SNAPSHOT TYPES
// =============================================================================

/// Validated durable UUID snapshot for a TDS topology.
///
/// This proof-bearing type is the internal boundary between raw interchange
/// records and live slotmap-backed runtime storage. Its private fields carry
/// UUID relationships that have already been checked for duplicate identities,
/// missing relationship records, dangling UUID references, and malformed
/// per-simplex arity.
#[derive(Debug)]
struct TdsSnapshot<U, V, const D: usize> {
    vertices: Vec<Vertex<U, D>>,
    simplices: Vec<TdsSnapshotSimplex<V, D>>,
}

/// Validated durable simplex identity and UUID connectivity.
///
/// Runtime `Simplex` values keep slotmap-local vertex and neighbor keys. This
/// snapshot simplex keeps the same relationships as UUIDs so hydration can
/// allocate fresh keys without trusting process-local handles from disk.
#[derive(Debug)]
struct TdsSnapshotSimplex<V, const D: usize> {
    uuid: Uuid,
    data: Option<V>,
    vertex_uuids: SnapshotVertexUuidSlots<D>,
    neighbor_uuids: SnapshotNeighborUuidSlots<D>,
    periodic_vertex_offsets: Option<SnapshotPeriodicOffsetSlots<D>>,
}

/// Validated UUID slots aligned with one D-simplex's vertex positions.
///
/// The raw interchange shape uses `Vec<Uuid>` because codecs cannot express the
/// `D + 1` invariant. This private wrapper is the parsed representation: it is
/// constructed only after checking arity, dangling UUID references, and duplicate
/// vertex UUIDs.
#[derive(Debug)]
struct SnapshotVertexUuidSlots<const D: usize> {
    slots: SimplexVertexUuidBuffer,
}

impl<const D: usize> SnapshotVertexUuidSlots<D> {
    /// Parses untrusted vertex UUID slots and proves they match one D-simplex.
    fn parse(
        simplex_uuid: Uuid,
        slots: &[Uuid],
        vertex_uuids: &SnapshotUuidSet,
    ) -> Result<Self, TdsSnapshotError> {
        validate_snapshot_vertex_slot_arity::<D>(simplex_uuid, slots.len())?;

        let mut seen_vertex_uuids = fast_hash_set_with_capacity(slots.len());
        for &vertex_uuid in slots {
            if !vertex_uuids.contains(&vertex_uuid) {
                return Err(TdsSnapshotError::DanglingSimplexVertexUuid {
                    simplex_uuid,
                    vertex_uuid,
                });
            }
            if !seen_vertex_uuids.insert(vertex_uuid) {
                return Err(TdsSnapshotError::InvalidSimplex {
                    simplex_uuid,
                    source: SimplexValidationError::DuplicateVertices,
                });
            }
        }

        Ok(Self::from_checked_slice(slots))
    }

    /// Builds vertex UUID slots from validated runtime state before serialization.
    fn from_runtime(simplex_uuid: Uuid, slots: &[Uuid]) -> Result<Self, TdsSnapshotError> {
        validate_snapshot_vertex_slot_arity::<D>(simplex_uuid, slots.len())?;

        let mut seen_vertex_uuids = fast_hash_set_with_capacity(slots.len());
        for &vertex_uuid in slots {
            if !seen_vertex_uuids.insert(vertex_uuid) {
                return Err(TdsSnapshotError::InvalidSimplex {
                    simplex_uuid,
                    source: SimplexValidationError::DuplicateVertices,
                });
            }
        }

        Ok(Self::from_checked_slice(slots))
    }

    /// Stores an already checked vertex UUID slice in the snapshot buffer type.
    fn from_checked_slice(slots: &[Uuid]) -> Self {
        let mut checked_slots = SimplexVertexUuidBuffer::with_capacity(slots.len());
        checked_slots.extend(slots.iter().copied());
        Self {
            slots: checked_slots,
        }
    }

    /// Exposes checked vertex UUID slots without reopening raw parse validation.
    fn iter(&self) -> impl Iterator<Item = &Uuid> {
        self.slots.iter()
    }

    /// Converts checked vertex UUID slots back to the raw codec-friendly shape.
    fn into_vec(self) -> Vec<Uuid> {
        self.slots.into_vec()
    }
}

/// Validated optional neighbor UUID slots aligned with one D-simplex's facets.
#[derive(Debug)]
struct SnapshotNeighborUuidSlots<const D: usize> {
    slots: NeighborBuffer<Option<Uuid>>,
}

impl<const D: usize> SnapshotNeighborUuidSlots<D> {
    /// Parses untrusted neighbor UUID slots and proves referenced simplices exist.
    fn parse(
        simplex_uuid: Uuid,
        slots: &[Option<Uuid>],
        simplex_uuids: &SnapshotUuidSet,
    ) -> Result<Self, TdsSnapshotError> {
        validate_snapshot_neighbor_slot_arity::<D>(simplex_uuid, slots.len())?;

        for &neighbor_uuid in slots.iter().flatten() {
            if !simplex_uuids.contains(&neighbor_uuid) {
                return Err(TdsSnapshotError::DanglingSimplexNeighborUuid {
                    simplex_uuid,
                    neighbor_uuid,
                });
            }
        }

        Ok(Self::from_checked_slice(slots))
    }

    /// Builds neighbor UUID slots from validated runtime neighbor keys.
    fn from_runtime(simplex_uuid: Uuid, slots: &[Option<Uuid>]) -> Result<Self, TdsSnapshotError> {
        validate_snapshot_neighbor_slot_arity::<D>(simplex_uuid, slots.len())?;
        Ok(Self::from_checked_slice(slots))
    }

    /// Stores an already checked neighbor UUID slice in the snapshot buffer type.
    fn from_checked_slice(slots: &[Option<Uuid>]) -> Self {
        let mut checked_slots = NeighborBuffer::with_capacity(slots.len());
        checked_slots.extend(slots.iter().copied());
        Self {
            slots: checked_slots,
        }
    }

    /// Exposes checked neighbor UUID slots for hydration into slotmap keys.
    fn iter(&self) -> impl Iterator<Item = &Option<Uuid>> {
        self.slots.iter()
    }

    /// Converts checked neighbor UUID slots back to the raw codec-friendly shape.
    fn into_vec(self) -> Vec<Option<Uuid>> {
        self.slots.into_vec()
    }
}

/// Validated periodic-offset slots aligned with one D-simplex's vertex slots.
#[derive(Debug)]
struct SnapshotPeriodicOffsetSlots<const D: usize> {
    slots: PeriodicOffsetBuffer<D>,
}

impl<const D: usize> SnapshotPeriodicOffsetSlots<D> {
    /// Parses raw periodic-offset rows and proves they match simplex vertex slots.
    fn parse(simplex_uuid: Uuid, offsets: &[Vec<i8>]) -> Result<Self, TdsSnapshotError> {
        let mut slots = PeriodicOffsetBuffer::new();
        for (offset_index, offset) in offsets.iter().enumerate() {
            let parsed_offset = offset.as_slice().try_into().map_err(|_| {
                TdsSnapshotError::PeriodicOffsetDimensionMismatch {
                    simplex_uuid,
                    offset_index,
                    expected: D,
                    actual: offset.len(),
                }
            })?;
            slots.push(parsed_offset);
        }
        Self::from_parsed_offsets(simplex_uuid, slots)
    }

    /// Builds periodic-offset slots from runtime fixed-size offset arrays.
    fn from_runtime(simplex_uuid: Uuid, offsets: &[[i8; D]]) -> Result<Self, TdsSnapshotError> {
        Self::from_parsed_offsets(simplex_uuid, offsets.iter().copied())
    }

    /// Stores parsed offsets after proving there is one offset per simplex vertex.
    fn from_parsed_offsets(
        simplex_uuid: Uuid,
        offsets: impl IntoIterator<Item = [i8; D]>,
    ) -> Result<Self, TdsSnapshotError> {
        let mut slots = PeriodicOffsetBuffer::new();
        slots.extend(offsets);
        if slots.len() != D + 1 {
            return Err(TdsSnapshotError::InvalidSimplex {
                simplex_uuid,
                source: SimplexValidationError::PeriodicOffsetLengthMismatch {
                    expected: D + 1,
                    found: slots.len(),
                },
            });
        }
        Ok(Self { slots })
    }

    /// Converts checked offsets to the runtime buffer expected by `Simplex`.
    fn into_buffer(self) -> PeriodicOffsetBuffer<D> {
        self.slots
    }

    /// Converts checked offsets back to the raw row-based codec shape.
    fn into_raw_rows(self) -> Vec<Vec<i8>> {
        self.slots
            .into_iter()
            .map(|offset| offset.to_vec())
            .collect()
    }
}

impl<SnapshotData, const D: usize> TdsSnapshotSimplex<SnapshotData, D> {
    /// Builds validated UUID relationships around caller-selected snapshot payload data.
    ///
    /// The relationship checks are identical for owned and borrowed payloads, so
    /// this helper lets [`Tds`] serialization borrow `U`/`V` data while tests can
    /// still build owned raw snapshots for mutation.
    fn from_simplex_with_data<U, RuntimeData>(
        tds: &Tds<U, RuntimeData, D>,
        simplex: &Simplex<RuntimeData, D>,
        data: Option<SnapshotData>,
    ) -> Result<Self, TdsSnapshotError> {
        let simplex_uuid = simplex.uuid();
        let vertex_uuids = simplex.vertex_uuids(tds).map_err(|source| {
            TdsSnapshotError::SimplexVertexUuidResolutionFailed {
                simplex_uuid,
                source,
            }
        })?;
        let neighbor_uuids = simplex
            .neighbor_keys()
            .ok_or(TdsSnapshotError::MissingSimplexNeighborSlots { simplex_uuid })?
            .map(|neighbor_key| {
                neighbor_key
                    .map(|neighbor_key| {
                        tds.simplex_uuid_from_key(neighbor_key).ok_or(
                            TdsSnapshotError::DanglingRuntimeNeighborKey {
                                simplex_uuid,
                                neighbor_key,
                            },
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;
        let vertex_uuids = SnapshotVertexUuidSlots::from_runtime(simplex_uuid, &vertex_uuids)?;
        let neighbor_uuids =
            SnapshotNeighborUuidSlots::from_runtime(simplex_uuid, &neighbor_uuids)?;
        let periodic_vertex_offsets = simplex
            .periodic_vertex_offsets()
            .map(|offsets| SnapshotPeriodicOffsetSlots::from_runtime(simplex_uuid, offsets))
            .transpose()?;

        Ok(Self {
            uuid: simplex_uuid,
            data,
            neighbor_uuids,
            vertex_uuids,
            periodic_vertex_offsets,
        })
    }

    /// Converts this validated simplex relationship record back into the raw
    /// simplex record and raw relationship maps used by the interchange shape.
    fn into_raw_parts(self) -> RawSnapshotSimplexParts<SnapshotData> {
        let raw_simplex = RawSnapshotSimplex {
            uuid: self.uuid,
            data: self.data,
        };
        let raw_offsets = self
            .periodic_vertex_offsets
            .map(SnapshotPeriodicOffsetSlots::into_raw_rows);
        (
            raw_simplex,
            self.vertex_uuids.into_vec(),
            self.neighbor_uuids.into_vec(),
            raw_offsets,
        )
    }
}

impl<V> Serialize for RawSnapshotSimplex<V>
where
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let has_data = self.data.is_some();
        let field_count = if has_data { 2 } else { 1 };
        let mut state = serializer.serialize_struct("Simplex", field_count)?;
        state.serialize_field("uuid", &self.uuid)?;
        if has_data {
            state.serialize_field("data", &self.data)?;
        }
        state.end()
    }
}

impl<'de, V> Deserialize<'de> for RawSnapshotSimplex<V>
where
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        struct SnapshotSimplexVisitor<V> {
            _phantom: PhantomData<V>,
        }

        impl<'de, V> Visitor<'de> for SnapshotSimplexVisitor<V>
        where
            V: DataDeserialize,
        {
            type Value = RawSnapshotSimplex<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a TDS snapshot simplex record")
            }

            fn visit_map<A>(self, mut map: A) -> Result<RawSnapshotSimplex<V>, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut uuid = None;
                let mut data: Option<V> = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "uuid" => {
                            if uuid.is_some() {
                                return Err(de::Error::duplicate_field("uuid"));
                            }
                            uuid = Some(map.next_value()?);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        "vertices" | "neighbors" | "periodic_vertex_offsets" => {
                            return Err(de::Error::custom(format!(
                                "{key} is storage-local simplex state and must not be deserialized; deserialize Tds so UUID relationships can be reconstructed",
                            )));
                        }
                        _ => {
                            return Err(de::Error::custom(format!(
                                "unknown snapshot simplex field `{key}`, expected `uuid` or `data`"
                            )));
                        }
                    }
                }

                let uuid: Uuid = uuid.ok_or_else(|| de::Error::missing_field("uuid"))?;
                validate_uuid(&uuid)
                    .map_err(|source| de::Error::custom(format!("invalid uuid: {source}")))?;

                Ok(RawSnapshotSimplex { uuid, data })
            }
        }

        const FIELDS: &[&str] = &["uuid", "data"];
        deserializer.deserialize_struct(
            "RawSnapshotSimplex",
            FIELDS,
            SnapshotSimplexVisitor {
                _phantom: PhantomData,
            },
        )
    }
}

struct ParsedVertexStorage<U, const D: usize> {
    vertices: StorageMap<VertexKey, Vertex<U, D>>,
    uuid_to_vertex_key: UuidToVertexKeyMap,
}

struct ParsedSimplexStorage<V, const D: usize> {
    simplices: StorageMap<SimplexKey, Simplex<V, D>>,
    uuid_to_simplex_key: UuidToSimplexKeyMap,
}

type SnapshotUuidSet = FastHashSet<Uuid>;
type SnapshotNeighborAssignments<const D: usize> =
    Vec<(Uuid, SimplexKey, SnapshotNeighborUuidSlots<D>)>;
type RawSnapshotSimplexParts<V> = (
    RawSnapshotSimplex<V>,
    Vec<Uuid>,
    Vec<Option<Uuid>>,
    Option<Vec<Vec<i8>>>,
);

// =============================================================================
// SNAPSHOT CONVERSION
// =============================================================================

impl<U, V, const D: usize> RawTdsSnapshot<U, V, D> {
    /// Parses raw snapshot records into a validated UUID snapshot.
    fn parse(self) -> Result<TdsSnapshot<U, V, D>, TdsSnapshotError> {
        let Self {
            vertices,
            simplices,
            simplex_vertices,
            simplex_neighbors,
            simplex_vertex_offsets,
        } = self;

        let vertex_uuids = collect_vertex_uuids(&vertices)?;
        let simplex_uuids = collect_simplex_uuids(&simplices)?;
        validate_relationship_keys(
            &simplex_uuids,
            &simplex_vertices,
            &simplex_neighbors,
            &simplex_vertex_offsets,
        )?;

        let simplices = simplices
            .into_iter()
            .map(|raw_simplex| {
                parse_raw_simplex(
                    raw_simplex,
                    &vertex_uuids,
                    &simplex_uuids,
                    &simplex_vertices,
                    &simplex_neighbors,
                    &simplex_vertex_offsets,
                )
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;

        Ok(TdsSnapshot {
            vertices,
            simplices,
        })
    }
}

impl<U, V, const D: usize> TdsSnapshot<U, V, D> {
    /// Converts a valid live key-based TDS into an owned snapshot for mutation tests.
    ///
    /// Production serialization uses the borrowed `from_tds` path below so non-`Copy`
    /// payloads can still cross the public [`Tds`] codec boundary.
    #[cfg(test)]
    fn from_tds_owned(tds: &Tds<U, V, D>) -> Result<Self, TdsSnapshotError>
    where
        U: Copy,
        V: Copy,
    {
        tds.validate()
            .map_err(|source| TdsSnapshotError::SourceValidationFailed { source })?;

        let vertices = tds
            .vertices()
            .map(|(_vertex_key, vertex)| *vertex)
            .collect();
        let simplices = tds
            .simplices()
            .map(|(_simplex_key, simplex)| {
                TdsSnapshotSimplex::from_simplex_with_data(tds, simplex, simplex.data)
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;

        Ok(Self {
            vertices,
            simplices,
        })
    }

    /// Converts this validated snapshot into the raw serializable shape.
    fn into_raw(self) -> RawTdsSnapshot<U, V, D> {
        let mut raw_simplices = Vec::with_capacity(self.simplices.len());
        let mut simplex_vertices = fast_hash_map_with_capacity(self.simplices.len());
        let mut simplex_neighbors = fast_hash_map_with_capacity(self.simplices.len());
        let mut simplex_vertex_offsets = fast_hash_map_with_capacity(self.simplices.len());

        for simplex in self.simplices {
            let simplex_uuid = simplex.uuid;
            let (raw_simplex, vertex_uuids, neighbor_uuids, offsets) = simplex.into_raw_parts();
            raw_simplices.push(raw_simplex);
            simplex_vertices.insert(simplex_uuid, vertex_uuids);
            simplex_neighbors.insert(simplex_uuid, neighbor_uuids);
            if let Some(offsets) = offsets {
                simplex_vertex_offsets.insert(simplex_uuid, offsets);
            }
        }

        RawTdsSnapshot {
            vertices: self.vertices,
            simplices: raw_simplices,
            simplex_vertices,
            simplex_neighbors,
            simplex_vertex_offsets,
        }
    }

    /// Parses a durable UUID snapshot into a fresh key-based runtime TDS.
    fn into_tds(self) -> Result<Tds<U, V, D>, TdsSnapshotError> {
        let ParsedVertexStorage {
            vertices,
            uuid_to_vertex_key,
        } = rebuild_vertices(self.vertices);

        let ParsedSimplexStorage {
            simplices,
            uuid_to_simplex_key,
        } = rebuild_simplices(&uuid_to_vertex_key, self.simplices)?;

        let mut tds = Tds {
            vertices,
            simplices,
            uuid_to_vertex_key,
            uuid_to_simplex_key,
            construction_state: TriangulationConstructionState::Constructed,
            generation: Arc::new(AtomicU64::new(0)),
            identity: Arc::new(Uuid::new_v4()),
        };

        tds.assign_incident_simplices()
            .map_err(|source| TdsSnapshotError::IncidentSimplexRebuildFailed { source })?;
        tds.validate()
            .map_err(|source| TdsSnapshotError::ValidationFailed { source })?;

        Ok(tds)
    }
}

impl<'a, U, V, const D: usize> TdsSnapshot<&'a U, &'a V, D> {
    /// Converts a valid live key-based TDS into a borrowed durable UUID snapshot.
    ///
    /// This is the production serialization path for [`Tds`]. It validates the
    /// live topology, stores UUID relationships, and borrows payload data so
    /// callers only need [`DataSerialize`] rather than `Copy`.
    fn from_tds(tds: &'a Tds<U, V, D>) -> Result<Self, TdsSnapshotError> {
        tds.validate()
            .map_err(|source| TdsSnapshotError::SourceValidationFailed { source })?;

        let vertices = tds
            .vertices()
            .map(|(_vertex_key, vertex)| {
                let mut snapshot_vertex =
                    Vertex::new_with_uuid(*vertex.point(), vertex.uuid(), vertex.data());
                snapshot_vertex.set_incident_simplex(vertex.incident_simplex());
                snapshot_vertex
            })
            .collect();
        let simplices = tds
            .simplices()
            .map(|(_simplex_key, simplex)| {
                TdsSnapshotSimplex::from_simplex_with_data(tds, simplex, simplex.data())
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;

        Ok(Self {
            vertices,
            simplices,
        })
    }
}

/// Rebuilds vertex storage and UUID mappings from validated snapshot vertex records.
///
/// This consumes the `TdsSnapshot` proof that vertex UUIDs are unique. Raw input
/// must pass through `RawTdsSnapshot::parse` before reaching this hydration step.
fn rebuild_vertices<U, const D: usize>(
    snapshot_vertices: Vec<Vertex<U, D>>,
) -> ParsedVertexStorage<U, D> {
    let mut vertices = StorageMap::with_capacity_and_key(snapshot_vertices.len());
    let mut uuid_to_vertex_key = fast_hash_map_with_capacity(snapshot_vertices.len());

    for vertex in snapshot_vertices {
        let vertex_uuid = vertex.uuid();
        let vertex_key = vertices.insert(vertex);
        uuid_to_vertex_key.insert(vertex_uuid, vertex_key);
    }

    ParsedVertexStorage {
        vertices,
        uuid_to_vertex_key,
    }
}

/// Collects vertex UUIDs from raw snapshot vertices and rejects duplicates.
fn collect_vertex_uuids<U, const D: usize>(
    snapshot_vertices: &[Vertex<U, D>],
) -> Result<SnapshotUuidSet, TdsSnapshotError> {
    let mut vertex_uuids = fast_hash_set_with_capacity(snapshot_vertices.len());
    for vertex in snapshot_vertices {
        let vertex_uuid = vertex.uuid();
        if !vertex_uuids.insert(vertex_uuid) {
            return Err(TdsSnapshotError::DuplicateVertexUuid { vertex_uuid });
        }
    }
    Ok(vertex_uuids)
}

/// Collects simplex UUIDs from raw snapshot simplex records and rejects duplicates.
fn collect_simplex_uuids<V>(
    snapshot_simplices: &[RawSnapshotSimplex<V>],
) -> Result<SnapshotUuidSet, TdsSnapshotError> {
    let mut simplex_uuids = fast_hash_set_with_capacity(snapshot_simplices.len());
    for simplex in snapshot_simplices {
        let simplex_uuid = simplex.uuid;
        if !simplex_uuids.insert(simplex_uuid) {
            return Err(TdsSnapshotError::DuplicateSimplexUuid { simplex_uuid });
        }
    }
    Ok(simplex_uuids)
}

/// Parses one raw simplex record into a validated UUID connectivity record.
fn parse_raw_simplex<V, const D: usize>(
    raw_simplex: RawSnapshotSimplex<V>,
    vertex_uuids: &SnapshotUuidSet,
    simplex_uuids: &SnapshotUuidSet,
    simplex_vertices: &FastHashMap<Uuid, Vec<Uuid>>,
    simplex_neighbors: &FastHashMap<Uuid, Vec<Option<Uuid>>>,
    simplex_vertex_offsets: &FastHashMap<Uuid, Vec<Vec<i8>>>,
) -> Result<TdsSnapshotSimplex<V, D>, TdsSnapshotError> {
    let simplex_uuid = raw_simplex.uuid;
    let data = raw_simplex.data;
    let vertex_uuid_slots = simplex_vertices
        .get(&simplex_uuid)
        .ok_or(TdsSnapshotError::MissingSimplexVertexUuids { simplex_uuid })?;
    let vertex_uuids =
        SnapshotVertexUuidSlots::parse(simplex_uuid, vertex_uuid_slots, vertex_uuids)?;

    let neighbor_uuid_slots = simplex_neighbors
        .get(&simplex_uuid)
        .ok_or(TdsSnapshotError::MissingSimplexNeighborUuids { simplex_uuid })?;
    let neighbor_uuids =
        SnapshotNeighborUuidSlots::parse(simplex_uuid, neighbor_uuid_slots, simplex_uuids)?;

    let periodic_vertex_offsets = simplex_vertex_offsets
        .get(&simplex_uuid)
        .map(|offsets| SnapshotPeriodicOffsetSlots::parse(simplex_uuid, offsets))
        .transpose()?;

    Ok(TdsSnapshotSimplex {
        uuid: simplex_uuid,
        data,
        vertex_uuids,
        neighbor_uuids,
        periodic_vertex_offsets,
    })
}

/// Validates the vertex UUID slot count for one parsed snapshot simplex.
fn validate_snapshot_vertex_slot_arity<const D: usize>(
    simplex_uuid: Uuid,
    actual: usize,
) -> Result<(), TdsSnapshotError> {
    if actual != D + 1 {
        return Err(TdsSnapshotError::InvalidSimplexVertexUuidSlotCount {
            simplex_uuid,
            actual,
            expected: D + 1,
        });
    }
    Ok(())
}

/// Validates the neighbor UUID slot count for one parsed snapshot simplex.
fn validate_snapshot_neighbor_slot_arity<const D: usize>(
    simplex_uuid: Uuid,
    actual: usize,
) -> Result<(), TdsSnapshotError> {
    if actual != D + 1 {
        return Err(TdsSnapshotError::InvalidSimplex {
            simplex_uuid,
            source: SimplexValidationError::InvalidNeighborsLength {
                actual,
                expected: D + 1,
                dimension: D,
            },
        });
    }
    Ok(())
}

/// Rebuilds simplex storage from validated snapshot UUID connectivity maps.
///
/// This consumes the `TdsSnapshot` proof that simplex UUIDs are unique, every
/// simplex has `D + 1` vertex and neighbor slots, relationship maps are complete,
/// and all serialized UUID references are non-dangling.
fn rebuild_simplices<V, const D: usize>(
    uuid_to_vertex_key: &UuidToVertexKeyMap,
    snapshot_simplices: Vec<TdsSnapshotSimplex<V, D>>,
) -> Result<ParsedSimplexStorage<V, D>, TdsSnapshotError> {
    let mut simplices = StorageMap::with_capacity_and_key(snapshot_simplices.len());
    let mut uuid_to_simplex_key = fast_hash_map_with_capacity(snapshot_simplices.len());
    let mut snapshot_neighbor_assignments = Vec::with_capacity(snapshot_simplices.len());

    for snapshot_simplex in snapshot_simplices {
        let simplex_uuid = snapshot_simplex.uuid;
        let vertex_keys = snapshot_simplex
            .vertex_uuids
            .iter()
            .map(|&vertex_uuid| {
                uuid_to_vertex_key.get(&vertex_uuid).copied().ok_or(
                    TdsSnapshotError::DanglingSimplexVertexUuid {
                        simplex_uuid,
                        vertex_uuid,
                    },
                )
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;

        let mut simplex =
            Simplex::try_new_with_uuid(vertex_keys, simplex_uuid, snapshot_simplex.data).map_err(
                |source| TdsSnapshotError::InvalidSimplex {
                    simplex_uuid,
                    source,
                },
            )?;
        if let Some(offsets) = snapshot_simplex.periodic_vertex_offsets {
            simplex
                .set_periodic_vertex_offsets(offsets.into_buffer())
                .map_err(|source| TdsSnapshotError::InvalidSimplex {
                    simplex_uuid,
                    source,
                })?;
        }

        let simplex_key = simplices.insert(simplex);
        uuid_to_simplex_key.insert(simplex_uuid, simplex_key);
        snapshot_neighbor_assignments.push((
            simplex_uuid,
            simplex_key,
            snapshot_simplex.neighbor_uuids,
        ));
    }

    assign_neighbors(
        &mut simplices,
        &uuid_to_simplex_key,
        snapshot_neighbor_assignments,
    )?;

    Ok(ParsedSimplexStorage {
        simplices,
        uuid_to_simplex_key,
    })
}

/// Rejects relationship maps that mention simplices absent from the snapshot
/// simplex records.
fn validate_relationship_keys(
    simplex_uuids: &SnapshotUuidSet,
    simplex_vertices: &FastHashMap<Uuid, Vec<Uuid>>,
    simplex_neighbors: &FastHashMap<Uuid, Vec<Option<Uuid>>>,
    simplex_vertex_offsets: &FastHashMap<Uuid, Vec<Vec<i8>>>,
) -> Result<(), TdsSnapshotError> {
    for &simplex_uuid in simplex_vertices.keys() {
        if !simplex_uuids.contains(&simplex_uuid) {
            return Err(TdsSnapshotError::UnknownSimplexVertexMapping { simplex_uuid });
        }
    }
    for &simplex_uuid in simplex_neighbors.keys() {
        if !simplex_uuids.contains(&simplex_uuid) {
            return Err(TdsSnapshotError::UnknownSimplexNeighborMapping { simplex_uuid });
        }
    }
    for &simplex_uuid in simplex_vertex_offsets.keys() {
        if !simplex_uuids.contains(&simplex_uuid) {
            return Err(TdsSnapshotError::UnknownSimplexOffsetMapping { simplex_uuid });
        }
    }

    Ok(())
}

/// Resolves snapshot neighbor UUID slots to live simplex-key slots.
fn assign_neighbors<V, const D: usize>(
    simplices: &mut StorageMap<SimplexKey, Simplex<V, D>>,
    uuid_to_simplex_key: &UuidToSimplexKeyMap,
    snapshot_neighbor_assignments: SnapshotNeighborAssignments<D>,
) -> Result<(), TdsSnapshotError> {
    for (simplex_uuid, simplex_key, neighbor_uuids) in snapshot_neighbor_assignments {
        let neighbor_keys = neighbor_uuids
            .iter()
            .map(|&neighbor_uuid| {
                neighbor_uuid
                    .map(|neighbor_uuid| {
                        uuid_to_simplex_key.get(&neighbor_uuid).copied().ok_or(
                            TdsSnapshotError::DanglingSimplexNeighborUuid {
                                simplex_uuid,
                                neighbor_uuid,
                            },
                        )
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, TdsSnapshotError>>()?;

        let simplex = simplices
            .get_mut(simplex_key)
            .ok_or(TdsSnapshotError::UnknownSimplexNeighborMapping { simplex_uuid })?;
        simplex
            .set_neighbors_from_keys(neighbor_keys)
            .map_err(|source| TdsSnapshotError::InvalidSimplex {
                simplex_uuid,
                source,
            })?;
    }

    Ok(())
}

// =============================================================================
// TDS CODEC IMPLEMENTATIONS
// =============================================================================

impl<U, V, const D: usize> Serialize for Tds<U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        TdsSnapshot::from_tds(self)
            .map_err(serde::ser::Error::custom)?
            .into_raw()
            .serialize(serializer)
    }
}

impl<'de, U, V, const D: usize> Deserialize<'de> for Tds<U, V, D>
where
    U: DataDeserialize,
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        RawTdsSnapshot::<U, V, D>::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)?
            .into_tds()
            .map_err(de::Error::custom)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DelaunayTriangulation;
    use crate::core::simplex::SimplexValidationError;
    use crate::core::tds::TriangulationConstructionState;
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use slotmap::KeyData;
    use std::assert_matches;

    fn initial_simplex_vertices_3d() -> [Vertex<(), 3>; 4] {
        [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
        ]
    }

    fn periodic_offset_tds_2d() -> (Tds<(), (), 2>, Uuid, Vec<[i8; 2]>) {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<(), _>::try_new([0.0, 1.0]).unwrap())
            .unwrap();
        let offsets = vec![[0_i8, 0_i8], [1_i8, 0_i8], [0_i8, 1_i8]];
        let mut simplex = Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap();
        simplex
            .set_periodic_vertex_offsets(offsets.clone())
            .unwrap();
        let simplex_uuid = simplex.uuid();
        tds.insert_simplex_with_mapping(simplex).unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        (tds, simplex_uuid, offsets)
    }

    fn raw_snapshot_from_tds<U, V, const D: usize>(tds: &Tds<U, V, D>) -> RawTdsSnapshot<U, V, D>
    where
        U: Copy,
        V: Copy,
    {
        TdsSnapshot::from_tds_owned(tds)
            .expect("TDS should snapshot")
            .into_raw()
    }

    #[derive(Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
    struct NonCopyPayload {
        label: String,
    }

    /// Builds raw snapshot JSON with a duplicated object key in one relationship map.
    ///
    /// `serde_json::Value` cannot represent duplicate object keys, so this helper
    /// constructs the JSON text needed to exercise duplicate-key deserialization.
    fn snapshot_json_with_duplicate_relationship_key(field: &str) -> String {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;

        let duplicate_value = match field {
            "simplex_vertices" => serde_json::to_string(
                snapshot
                    .simplex_vertices
                    .get(&simplex_uuid)
                    .expect("snapshot should contain simplex vertex UUIDs"),
            ),
            "simplex_neighbors" => serde_json::to_string(
                snapshot
                    .simplex_neighbors
                    .get(&simplex_uuid)
                    .expect("snapshot should contain simplex neighbor UUIDs"),
            ),
            "simplex_vertex_offsets" => serde_json::to_string(&[[0_i8; 3]; 4]),
            _ => panic!("unknown relationship map field {field}"),
        }
        .expect("duplicate relationship value should serialize");
        let duplicate_map =
            format!(r#"{{"{simplex_uuid}":{duplicate_value},"{simplex_uuid}":{duplicate_value}}}"#);

        let vertices =
            serde_json::to_string(&snapshot.vertices).expect("snapshot vertices should serialize");
        let simplices = serde_json::to_string(&snapshot.simplices)
            .expect("snapshot simplices should serialize");
        let simplex_vertices = if field == "simplex_vertices" {
            duplicate_map.clone()
        } else {
            serde_json::to_string(&snapshot.simplex_vertices)
                .expect("simplex_vertices should serialize")
        };
        let simplex_neighbors = if field == "simplex_neighbors" {
            duplicate_map.clone()
        } else {
            serde_json::to_string(&snapshot.simplex_neighbors)
                .expect("simplex_neighbors should serialize")
        };
        let simplex_vertex_offsets = if field == "simplex_vertex_offsets" {
            duplicate_map
        } else {
            serde_json::to_string(&snapshot.simplex_vertex_offsets)
                .expect("simplex_vertex_offsets should serialize")
        };

        format!(
            r#"{{"vertices":{vertices},"simplices":{simplices},"simplex_vertices":{simplex_vertices},"simplex_neighbors":{simplex_neighbors},"simplex_vertex_offsets":{simplex_vertex_offsets}}}"#
        )
    }

    fn serialized_records<'a>(json: &'a serde_json::Value, field: &str) -> &'a [serde_json::Value] {
        let Some(records) = json.get(field).and_then(serde_json::Value::as_array) else {
            panic!("serialized TDS should contain {field} records");
        };
        let records = records.as_slice();

        for record in records {
            assert!(
                record.get("value").is_none(),
                "serialized TDS {field} records must not use slotmap value wrappers"
            );
        }

        records
    }

    fn serialized_uuid_field(json: &serde_json::Value, field: &str) -> Uuid {
        let uuid = json
            .get(field)
            .and_then(serde_json::Value::as_str)
            .and_then(|value| Uuid::parse_str(value).ok())
            .unwrap_or_else(|| panic!("serialized record should contain {field} UUID"));
        assert!(!uuid.is_nil(), "serialized {field} UUID should not be nil");
        uuid
    }

    #[test]
    fn test_tds_snapshot_serialization_includes_stable_uuid_relationships() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let original = dt.tds().clone();
        let json = serde_json::to_value(&original).expect("serialize TDS to JSON value");

        let vertex_uuids: Vec<_> = serialized_records(&json, "vertices")
            .iter()
            .map(|vertex| serialized_uuid_field(vertex, "uuid"))
            .collect();
        assert_eq!(vertex_uuids.len(), original.number_of_vertices());
        for (_, vertex) in original.vertices() {
            assert!(vertex_uuids.contains(&vertex.uuid()));
        }

        let simplex_uuids: Vec<_> = serialized_records(&json, "simplices")
            .iter()
            .map(|simplex| {
                assert!(
                    simplex.get("vertices").is_none(),
                    "serialized Simplex must not store slotmap VertexKey values"
                );
                serialized_uuid_field(simplex, "uuid")
            })
            .collect();
        assert_eq!(simplex_uuids.len(), original.number_of_simplices());
        for (_, simplex) in original.simplices() {
            assert!(simplex_uuids.contains(&simplex.uuid()));
        }

        let simplex_vertices = json
            .get("simplex_vertices")
            .and_then(serde_json::Value::as_object)
            .expect("serialized TDS should contain simplex_vertices object");
        assert_eq!(simplex_vertices.len(), original.number_of_simplices());

        for (simplex_uuid, serialized_vertex_uuids) in simplex_vertices {
            let simplex_uuid =
                Uuid::parse_str(simplex_uuid).expect("simplex_vertices keys should be UUIDs");
            assert!(simplex_uuids.contains(&simplex_uuid));
            let simplex_key = original
                .simplex_key_from_uuid(&simplex_uuid)
                .expect("serialized simplex UUID should resolve in original TDS");
            let expected_vertex_uuids = original
                .simplex(simplex_key)
                .expect("serialized simplex key should resolve")
                .vertex_uuids(&original)
                .expect("simplex vertex UUIDs should resolve");
            let serialized_vertex_uuids: Vec<_> = serialized_vertex_uuids
                .as_array()
                .expect("simplex_vertices values should be UUID arrays")
                .iter()
                .map(|vertex_uuid| {
                    let vertex_uuid = vertex_uuid
                        .as_str()
                        .and_then(|value| Uuid::parse_str(value).ok())
                        .expect("simplex vertex reference should be a UUID string");
                    assert!(vertex_uuids.contains(&vertex_uuid));
                    vertex_uuid
                })
                .collect();
            assert_eq!(
                serialized_vertex_uuids.as_slice(),
                expected_vertex_uuids.as_slice()
            );
        }

        let simplex_neighbors = json
            .get("simplex_neighbors")
            .and_then(serde_json::Value::as_object)
            .expect("serialized TDS should contain simplex_neighbors object");
        assert_eq!(simplex_neighbors.len(), original.number_of_simplices());

        for (simplex_uuid, serialized_neighbor_uuids) in simplex_neighbors {
            let simplex_uuid =
                Uuid::parse_str(simplex_uuid).expect("simplex_neighbors keys should be UUIDs");
            assert!(simplex_uuids.contains(&simplex_uuid));
            let simplex_key = original
                .simplex_key_from_uuid(&simplex_uuid)
                .expect("serialized simplex UUID should resolve in original TDS");
            let expected_neighbor_uuids: Vec<_> = original
                .simplex(simplex_key)
                .expect("serialized simplex key should resolve")
                .neighbors()
                .expect("serialized TDS should have assigned neighbor slots")
                .map(|neighbor_key| {
                    neighbor_key.and_then(|key| original.simplex_uuid_from_key(key))
                })
                .collect();
            let serialized_neighbor_uuids: Vec<_> = serialized_neighbor_uuids
                .as_array()
                .expect("simplex_neighbors values should be nullable UUID arrays")
                .iter()
                .map(|neighbor_uuid| {
                    neighbor_uuid.as_str().map(|value| {
                        Uuid::parse_str(value).expect("simplex neighbor reference should be a UUID")
                    })
                })
                .collect();
            assert_eq!(serialized_neighbor_uuids, expected_neighbor_uuids);
        }
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_unknown_simplex_record_fields() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut json = serde_json::to_value(dt.tds()).expect("serialize TDS to JSON value");

        let simplex_record = json
            .get_mut("simplices")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|simplices| simplices.first_mut())
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain a simplex record object");
        simplex_record.insert("unexpected".to_owned(), serde_json::json!(true));

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("unknown snapshot simplex fields should be rejected");

        assert!(
            err.to_string()
                .contains("unknown snapshot simplex field `unexpected`"),
            "unexpected error for unknown snapshot simplex field: {err}"
        );
    }

    #[test]
    fn test_raw_snapshot_simplex_rejects_duplicate_fields() {
        let uuid = Uuid::new_v4();
        let duplicate_uuid = Uuid::new_v4();
        let duplicate_uuid_json = format!(r#"{{"uuid":"{uuid}","uuid":"{duplicate_uuid}"}}"#);

        let err = serde_json::from_str::<RawSnapshotSimplex<()>>(&duplicate_uuid_json)
            .expect_err("duplicate raw simplex UUID fields should be rejected");
        assert!(
            err.to_string().contains("duplicate field `uuid`"),
            "unexpected error for duplicate raw simplex UUID field: {err}"
        );

        let duplicate_data_json = format!(r#"{{"uuid":"{uuid}","data":null,"data":null}}"#);

        let err = serde_json::from_str::<RawSnapshotSimplex<()>>(&duplicate_data_json)
            .expect_err("duplicate raw simplex data fields should be rejected");
        assert!(
            err.to_string().contains("duplicate field `data`"),
            "unexpected error for duplicate raw simplex data field: {err}"
        );
    }

    #[test]
    fn test_raw_snapshot_simplex_preserves_explicit_null_payload() {
        let uuid = Uuid::new_v4();
        let explicit_null_json = format!(r#"{{"uuid":"{uuid}","data":null}}"#);
        let raw = serde_json::from_str::<RawSnapshotSimplex<Option<i32>>>(&explicit_null_json)
            .expect("explicit null payload should deserialize");

        assert_eq!(raw.data, Some(None));

        let missing_data_json = format!(r#"{{"uuid":"{uuid}"}}"#);
        let raw = serde_json::from_str::<RawSnapshotSimplex<Option<i32>>>(&missing_data_json)
            .expect("missing payload should deserialize");

        assert_eq!(raw.data, None);
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_duplicate_relationship_map_keys() {
        for field in [
            "simplex_vertices",
            "simplex_neighbors",
            "simplex_vertex_offsets",
        ] {
            let json = snapshot_json_with_duplicate_relationship_key(field);
            let err = serde_json::from_str::<RawTdsSnapshot<(), (), 3>>(&json)
                .expect_err("duplicate relationship map keys should be rejected");
            let message = err.to_string();

            assert!(
                message.contains("duplicate simplex UUID key"),
                "unexpected error for duplicate {field} key: {err}"
            );
            assert!(
                message.contains(field),
                "duplicate relationship map error should identify {field}: {err}"
            );
        }
    }

    #[test]
    fn test_raw_snapshot_simplex_rejects_storage_local_fields() {
        let uuid = Uuid::new_v4();

        for field in ["vertices", "neighbors", "periodic_vertex_offsets"] {
            let err = serde_json::from_value::<RawSnapshotSimplex<()>>(serde_json::json!({
                "uuid": uuid,
                field: [],
            }))
            .expect_err("storage-local raw simplex fields should be rejected");

            assert!(
                err.to_string().contains("storage-local simplex state"),
                "unexpected error for storage-local raw simplex field {field}: {err}"
            );
        }
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_unknown_top_level_fields() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut json = serde_json::to_value(dt.tds()).expect("serialize TDS to JSON value");
        json.as_object_mut()
            .expect("serialized TDS should be an object")
            .insert("unexpected".to_owned(), serde_json::json!(true));

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("unknown top-level snapshot fields should be rejected");

        assert!(
            err.to_string().contains("unknown field `unexpected`"),
            "unexpected error for unknown top-level snapshot field: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_missing_top_level_neighbor_map() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut json = serde_json::to_value(dt.tds()).expect("serialize TDS to JSON value");
        json.as_object_mut()
            .expect("serialized TDS should be an object")
            .remove("simplex_neighbors")
            .expect("serialized TDS should contain simplex_neighbors");

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("missing top-level simplex_neighbors should be rejected");

        assert!(
            err.to_string()
                .contains("missing field `simplex_neighbors`"),
            "unexpected error for missing top-level simplex_neighbors: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_serde_round_trip_preserves_tds_structure() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let original = dt.tds().clone();

        let json = serde_json::to_string(&original).expect("serialize failed");
        let deserialized: Tds<(), (), 3> = serde_json::from_str(&json).expect("deserialize failed");

        assert_eq!(
            deserialized.number_of_vertices(),
            original.number_of_vertices()
        );
        assert_eq!(
            deserialized.number_of_simplices(),
            original.number_of_simplices()
        );
        assert_eq!(deserialized.dim(), original.dim());
        assert_eq!(deserialized, original);
        assert!(deserialized.is_valid().is_ok());
    }

    #[test]
    fn test_tds_snapshot_serde_round_trip_multi_simplex_triangulation() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let original = dt.tds().clone();
        assert!(original.number_of_simplices() > 1);

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Tds<(), (), 3> = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized, original);
        assert!(deserialized.is_valid().is_ok());
        assert!(deserialized.is_connected());
        assert!(deserialized.is_coherently_oriented());
    }

    #[test]
    fn test_tds_snapshot_serde_round_trip_2d() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::try_new(&vertices).unwrap();
        let original = dt.tds().clone();

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Tds<(), (), 2> = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized, original);
        assert!(deserialized.is_valid().is_ok());
    }

    #[test]
    fn test_tds_snapshot_serde_round_trip_preserves_periodic_offsets() {
        let (original, simplex_uuid, expected_offsets) = periodic_offset_tds_2d();
        let json = serde_json::to_value(&original).expect("serialize periodic-offset TDS");

        let offset_records = json
            .get("simplex_vertex_offsets")
            .and_then(serde_json::Value::as_object)
            .expect("serialized TDS should contain periodic offset map");
        let serialized_offsets = offset_records
            .get(&simplex_uuid.to_string())
            .and_then(serde_json::Value::as_array)
            .expect("serialized offset map should contain the simplex UUID");
        assert_eq!(serialized_offsets.len(), expected_offsets.len());

        let deserialized: Tds<(), (), 2> =
            serde_json::from_value(json).expect("deserialize periodic-offset TDS");
        let deserialized_simplex_key = deserialized
            .simplex_key_from_uuid(&simplex_uuid)
            .expect("simplex UUID should resolve after round-trip");
        let restored_offsets = deserialized
            .simplex(deserialized_simplex_key)
            .and_then(Simplex::periodic_vertex_offsets)
            .expect("periodic offsets should survive TDS serde round-trip");

        assert_eq!(restored_offsets, expected_offsets.as_slice());
        assert!(deserialized.is_valid().is_ok());
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_invalid_periodic_offset_mappings() {
        let (original, simplex_uuid, _expected_offsets) = periodic_offset_tds_2d();
        let json = serde_json::to_value(&original).expect("serialize periodic-offset TDS");

        let mut wrong_offset_count = json.clone();
        wrong_offset_count
            .get_mut("simplex_vertex_offsets")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain periodic offset map")
            .insert(simplex_uuid.to_string(), serde_json::json!([[0_i8, 0_i8]]));
        let err = serde_json::from_value::<Tds<(), (), 2>>(wrong_offset_count)
            .expect_err("wrong periodic offset count should be rejected");
        assert!(
            err.to_string().contains("Periodic offset length mismatch"),
            "unexpected error for wrong periodic offset count: {err}"
        );

        let mut wrong_offset_dimension = json.clone();
        wrong_offset_dimension
            .get_mut("simplex_vertex_offsets")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain periodic offset map")
            .insert(
                simplex_uuid.to_string(),
                serde_json::json!([[0_i8, 0_i8], [1_i8], [0_i8, 1_i8]]),
            );
        let err = serde_json::from_value::<Tds<(), (), 2>>(wrong_offset_dimension)
            .expect_err("wrong periodic offset dimension should be rejected");
        assert!(
            err.to_string().contains("has dimension 1, expected 2"),
            "unexpected error for wrong periodic offset dimension: {err}"
        );

        let mut unknown_simplex = json;
        unknown_simplex
            .get_mut("simplex_vertex_offsets")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain periodic offset map")
            .insert(
                Uuid::new_v4().to_string(),
                serde_json::json!([[0_i8, 0_i8], [1_i8, 0_i8], [0_i8, 1_i8]]),
            );
        let err = serde_json::from_value::<Tds<(), (), 2>>(unknown_simplex)
            .expect_err("unknown periodic-offset simplex UUID should be rejected");
        assert!(
            err.to_string().contains("unknown simplex"),
            "unexpected error for unknown periodic-offset simplex UUID: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_duplicate_vertex_uuids() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let original = dt.tds().clone();
        let mut json = serde_json::to_value(&original).expect("serialize TDS to JSON value");

        let vertex_records = json
            .get_mut("vertices")
            .and_then(serde_json::Value::as_array_mut)
            .expect("serialized TDS should contain vertex records");
        assert!(
            vertex_records
                .iter()
                .all(|record| record.get("value").is_none()),
            "serialized vertices must not use slotmap value wrappers"
        );
        let mut populated_vertices = vertex_records.iter_mut();
        let first_uuid = populated_vertices
            .next()
            .and_then(|vertex| vertex.get("uuid"))
            .cloned()
            .expect("first serialized vertex should contain uuid");
        populated_vertices
            .next()
            .and_then(|vertex| vertex.get_mut("uuid"))
            .map(|uuid| *uuid = first_uuid)
            .expect("second serialized vertex should contain uuid");

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("duplicate serialized vertex UUIDs should be rejected");
        assert!(
            err.to_string().contains("Duplicate vertex UUID"),
            "unexpected error for duplicate vertex UUIDs: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_extra_simplex_vertex_uuid_mapping() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let original = dt.tds().clone();
        let mut json = serde_json::to_value(&original).expect("serialize TDS to JSON value");
        let (_, simplex) = original
            .simplices()
            .next()
            .expect("single tetrahedron should have a simplex");
        let vertex_uuids = simplex
            .vertex_uuids(&original)
            .expect("simplex vertices should resolve");
        let unknown_simplex_uuid = Uuid::new_v4();

        json.get_mut("simplex_vertices")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain simplex_vertices")
            .insert(
                unknown_simplex_uuid.to_string(),
                serde_json::json!(vertex_uuids.to_vec()),
            );

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("extra simplex UUID mapping should be rejected");
        assert!(
            err.to_string().contains("unknown simplex"),
            "unexpected error for extra simplex UUID mapping: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_unknown_relationship_map_simplex_uuids() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();

        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let unknown_neighbor_simplex_uuid = Uuid::new_v4();
        snapshot
            .simplex_neighbors
            .insert(unknown_neighbor_simplex_uuid, vec![None, None, None, None]);

        let err = snapshot
            .parse()
            .expect_err("unknown simplex neighbor mapping should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::UnknownSimplexNeighborMapping { simplex_uuid }
                if simplex_uuid == unknown_neighbor_simplex_uuid
        );

        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let unknown_offset_simplex_uuid = Uuid::new_v4();
        snapshot
            .simplex_vertex_offsets
            .insert(unknown_offset_simplex_uuid, vec![vec![0_i8, 0_i8, 0_i8]; 4]);

        let err = snapshot
            .parse()
            .expect_err("unknown simplex offset mapping should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::UnknownSimplexOffsetMapping { simplex_uuid }
                if simplex_uuid == unknown_offset_simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_invalid_simplex_vertex_uuid_mappings() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let original = dt.tds().clone();
        let json = serde_json::to_value(&original).expect("serialize TDS to JSON value");
        let (_, simplex) = original
            .simplices()
            .next()
            .expect("single tetrahedron should have a simplex");
        let simplex_uuid = simplex.uuid();
        let vertex_uuids = simplex
            .vertex_uuids(&original)
            .expect("simplex vertices should resolve");

        let mut too_few_vertices = json.clone();
        too_few_vertices
            .get_mut("simplex_vertices")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain simplex_vertices")
            .insert(
                simplex_uuid.to_string(),
                serde_json::json!([vertex_uuids[0]]),
            );
        let err = serde_json::from_value::<Tds<(), (), 3>>(too_few_vertices)
            .expect_err("wrong simplex vertex count should be rejected");
        assert!(
            err.to_string().contains("vertex UUID slots"),
            "unexpected error for wrong vertex count: {err}"
        );

        let mut duplicate_vertex = json.clone();
        duplicate_vertex
            .get_mut("simplex_vertices")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain simplex_vertices")
            .insert(
                simplex_uuid.to_string(),
                serde_json::json!([
                    vertex_uuids[0],
                    vertex_uuids[0],
                    vertex_uuids[0],
                    vertex_uuids[0]
                ]),
            );
        let err = serde_json::from_value::<Tds<(), (), 3>>(duplicate_vertex)
            .expect_err("duplicate simplex vertex UUIDs should be rejected");
        assert!(
            err.to_string().contains("Duplicate vertices"),
            "unexpected error for duplicate vertex UUIDs: {err}"
        );

        let mut unknown_vertex = json;
        let unknown_uuid = Uuid::new_v4();
        unknown_vertex
            .get_mut("simplex_vertices")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain simplex_vertices")
            .insert(
                simplex_uuid.to_string(),
                serde_json::json!([
                    vertex_uuids[0],
                    vertex_uuids[1],
                    vertex_uuids[2],
                    unknown_uuid
                ]),
            );
        let err = serde_json::from_value::<Tds<(), (), 3>>(unknown_vertex)
            .expect_err("unknown vertex UUID should be rejected");
        assert!(
            err.to_string().contains("not found in vertices"),
            "unexpected error for unknown vertex UUID: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_deserialize_rejects_duplicate_simplex_vertex_uuid_sets() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let original = dt.tds().clone();
        assert!(original.number_of_simplices() > 1);
        let mut json = serde_json::to_value(&original).expect("serialize TDS to JSON value");

        let simplex_uuids: Vec<_> = serialized_records(&json, "simplices")
            .iter()
            .map(|simplex| serialized_uuid_field(simplex, "uuid"))
            .collect();
        let [first_simplex_uuid, second_simplex_uuid, ..] = simplex_uuids.as_slice() else {
            panic!("test triangulation should serialize at least two simplices");
        };
        let simplex_vertices = json
            .get_mut("simplex_vertices")
            .and_then(serde_json::Value::as_object_mut)
            .expect("serialized TDS should contain simplex_vertices");
        let duplicated_vertices = simplex_vertices
            .get(&first_simplex_uuid.to_string())
            .cloned()
            .expect("first simplex should have serialized vertex UUIDs");
        simplex_vertices.insert(second_simplex_uuid.to_string(), duplicated_vertices);

        let err = serde_json::from_value::<Tds<(), (), 3>>(json)
            .expect_err("duplicate simplex vertex UUID sets should be rejected");
        let error_message = err.to_string();
        assert!(
            error_message.contains("Duplicate simplices")
                || error_message.contains("Facet with key")
                || error_message.contains("2-manifold"),
            "unexpected error for duplicate simplex vertex UUID sets: {err}"
        );
    }

    #[test]
    fn test_tds_snapshot_round_trips_vertex_and_simplex_payload_data() {
        let mut tds: Tds<i32, i32, 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(Vertex::<_, _>::try_new_with_data([0.0, 0.0], 10).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(Vertex::<_, _>::try_new_with_data([1.0, 0.0], 20).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(Vertex::<_, _>::try_new_with_data([0.0, 1.0], 30).unwrap())
            .unwrap();
        let simplex = Simplex::try_new_with_data(vec![v0, v1, v2], Some(99)).unwrap();

        tds.insert_simplex_with_mapping(simplex).unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        let json = serde_json::to_string(&tds).expect("TDS snapshot should serialize");
        let restored: Tds<i32, i32, 2> =
            serde_json::from_str(&json).expect("TDS snapshot should deserialize");

        restored.validate().expect("restored TDS should be valid");
        let mut vertex_data = restored
            .vertices()
            .map(|(_vertex_key, vertex)| vertex.data().copied())
            .collect::<Vec<_>>();
        vertex_data.sort_unstable();
        let simplex_data = restored
            .simplices()
            .map(|(_simplex_key, simplex)| simplex.data().copied())
            .collect::<Vec<_>>();

        assert_eq!(vertex_data, vec![Some(10), Some(20), Some(30)]);
        assert_eq!(simplex_data, vec![Some(99)]);
    }

    #[test]
    fn test_tds_snapshot_serializes_non_copy_payload_data() {
        let mut tds: Tds<NonCopyPayload, NonCopyPayload, 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(
                Vertex::<_, _>::try_new_with_data(
                    [0.0, 0.0],
                    NonCopyPayload {
                        label: "v0".to_owned(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(
                Vertex::<_, _>::try_new_with_data(
                    [1.0, 0.0],
                    NonCopyPayload {
                        label: "v1".to_owned(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(
                Vertex::<_, _>::try_new_with_data(
                    [0.0, 1.0],
                    NonCopyPayload {
                        label: "v2".to_owned(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let simplex = Simplex::try_new_with_data(
            vec![v0, v1, v2],
            Some(NonCopyPayload {
                label: "simplex".to_owned(),
            }),
        )
        .unwrap();

        tds.insert_simplex_with_mapping(simplex).unwrap();
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();

        let json = serde_json::to_string(&tds).expect("TDS snapshot should serialize");
        let restored: Tds<NonCopyPayload, NonCopyPayload, 2> =
            serde_json::from_str(&json).expect("TDS snapshot should deserialize");

        restored.validate().expect("restored TDS should be valid");
        let mut vertex_data = restored
            .vertices()
            .filter_map(|(_vertex_key, vertex)| vertex.data().map(|payload| payload.label.as_str()))
            .collect::<Vec<_>>();
        vertex_data.sort_unstable();
        let simplex_data = restored
            .simplices()
            .filter_map(|(_simplex_key, simplex)| {
                simplex.data().map(|payload| payload.label.as_str())
            })
            .collect::<Vec<_>>();

        assert_eq!(vertex_data, vec!["v0", "v1", "v2"]);
        assert_eq!(simplex_data, vec!["simplex"]);
    }

    #[test]
    fn test_tds_snapshot_rejects_missing_runtime_neighbor_slots() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut tds = dt.tds().clone();
        let simplex_uuid = tds
            .simplices()
            .next()
            .map(|(_simplex_key, simplex)| simplex)
            .expect("test TDS should contain a simplex")
            .uuid();
        tds.clear_all_neighbors();

        let err = TdsSnapshot::from_tds(&tds)
            .expect_err("snapshotting a TDS without assigned neighbors should fail");

        assert_matches!(
            err,
            TdsSnapshotError::MissingSimplexNeighborSlots { simplex_uuid: found }
                if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_duplicate_vertex_uuid() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());

        let duplicate_uuid = snapshot
            .vertices
            .first()
            .expect("snapshot should contain vertices")
            .uuid();
        let duplicate_point = *snapshot
            .vertices
            .get(1)
            .expect("snapshot should contain at least two vertices")
            .point();
        snapshot.vertices[1] = Vertex::try_new_with_uuid(duplicate_point, duplicate_uuid, None)
            .expect("duplicate UUID fixture should still be a valid vertex record");

        let err = snapshot
            .parse()
            .expect_err("duplicate snapshot vertex UUIDs should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::DuplicateVertexUuid { vertex_uuid }
                if vertex_uuid == duplicate_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_hydration_relies_on_validated_vertex_uuid_uniqueness() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds())
            .parse()
            .expect("raw snapshot should parse into a validated snapshot");

        let duplicate_uuid = snapshot
            .vertices
            .first()
            .expect("snapshot should contain vertices")
            .uuid();
        let duplicate_point = *snapshot
            .vertices
            .get(1)
            .expect("snapshot should contain at least two vertices")
            .point();
        snapshot.vertices[1] = Vertex::try_new_with_uuid(duplicate_point, duplicate_uuid, None)
            .expect("duplicate UUID fixture should still be a valid vertex record");

        let err = snapshot
            .into_tds()
            .expect_err("invalid internal snapshot proof should fail during hydration");

        assert_matches!(err, TdsSnapshotError::DanglingSimplexVertexUuid { .. });
    }

    #[test]
    fn test_tds_snapshot_error_preserves_missing_simplex_vertex_mapping() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;

        snapshot.simplex_vertices.remove(&simplex_uuid);

        let err = snapshot
            .parse()
            .expect_err("missing simplex vertex UUID mapping should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::MissingSimplexVertexUuids { simplex_uuid: found }
                if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_invalid_simplex_vertex_uuid_slot_count() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;
        let vertex_uuids = snapshot
            .simplex_vertices
            .get_mut(&simplex_uuid)
            .expect("simplex should have snapshot vertex UUIDs");

        vertex_uuids.push(Uuid::new_v4());

        let err = snapshot
            .parse()
            .expect_err("too many simplex vertex UUID slots should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::InvalidSimplexVertexUuidSlotCount {
                simplex_uuid: found,
                actual: 5,
                expected: 4,
            } if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_missing_simplex_neighbor_mapping() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;

        snapshot.simplex_neighbors.remove(&simplex_uuid);

        let err = snapshot
            .parse()
            .expect_err("missing simplex neighbor UUID mapping should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::MissingSimplexNeighborUuids { simplex_uuid: found }
                if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_dangling_vertex_uuid_reference() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;
        let dangling_vertex_uuid = Uuid::new_v4();

        snapshot
            .simplex_vertices
            .get_mut(&simplex_uuid)
            .expect("simplex should have snapshot vertex UUIDs")[0] = dangling_vertex_uuid;

        let err = snapshot
            .parse()
            .expect_err("dangling simplex vertex UUID should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::DanglingSimplexVertexUuid {
                simplex_uuid: found_simplex,
                vertex_uuid: found_vertex,
            } if found_simplex == simplex_uuid && found_vertex == dangling_vertex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_dangling_neighbor_uuid_reference() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = *snapshot
            .simplex_neighbors
            .iter()
            .find_map(|(simplex_uuid, neighbors)| {
                neighbors
                    .iter()
                    .any(Option::is_some)
                    .then_some(simplex_uuid)
            })
            .expect("multi-simplex TDS should have an interior neighbor");
        let dangling_neighbor_uuid = Uuid::new_v4();

        let neighbors = snapshot
            .simplex_neighbors
            .get_mut(&simplex_uuid)
            .expect("simplex should have snapshot neighbor UUIDs");
        let interior_slot = neighbors
            .iter()
            .position(Option::is_some)
            .expect("simplex should have an interior neighbor");
        neighbors[interior_slot] = Some(dangling_neighbor_uuid);

        let err = snapshot
            .parse()
            .expect_err("dangling simplex neighbor UUID should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::DanglingSimplexNeighborUuid {
                simplex_uuid: found_simplex,
                neighbor_uuid: found_neighbor,
            } if found_simplex == simplex_uuid && found_neighbor == dangling_neighbor_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_periodic_offset_dimension_mismatch() {
        let (_original, simplex_uuid, _expected_offsets) = periodic_offset_tds_2d();
        let offsets = vec![vec![0_i8, 0_i8], vec![1_i8], vec![0_i8, 1_i8]];

        let err = SnapshotPeriodicOffsetSlots::<2>::parse(simplex_uuid, &offsets)
            .expect_err("wrong periodic offset dimension should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::PeriodicOffsetDimensionMismatch {
                simplex_uuid: found_simplex,
                offset_index: 1,
                expected: 2,
                actual: 1,
            } if found_simplex == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_duplicate_simplex_uuid() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;

        snapshot.simplices.push(RawSnapshotSimplex {
            uuid: simplex_uuid,
            data: None,
        });

        let err = snapshot
            .parse()
            .expect_err("duplicate snapshot simplex UUIDs should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::DuplicateSimplexUuid { simplex_uuid: found }
                if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_error_preserves_inconsistent_neighbor_connectivity() {
        let vertices = [
            Vertex::<(), _>::try_new([0.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([1.0, 0.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 1.0, 0.0]).unwrap(),
            Vertex::<(), _>::try_new([0.0, 0.0, 1.0]).unwrap(),
            Vertex::<(), _>::try_new([0.5, 0.5, 0.5]).unwrap(),
        ];
        let dt = DelaunayTriangulation::try_new(&vertices).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let (simplex_uuid, neighbors) = snapshot
            .simplex_neighbors
            .iter_mut()
            .find(|(_, neighbors)| neighbors.iter().any(Option::is_some))
            .expect("multi-simplex TDS should have an interior neighbor");
        let interior_slot = neighbors
            .iter()
            .position(Option::is_some)
            .expect("simplex should have an interior neighbor");
        neighbors[interior_slot] = None;
        let simplex_uuid = *simplex_uuid;

        let err = snapshot
            .parse()
            .expect("one-sided neighbor deletion is a structurally complete snapshot")
            .into_tds()
            .expect_err("one-sided neighbor deletion should be rejected");

        assert_matches!(err, TdsSnapshotError::ValidationFailed { .. });
        assert!(
            err.to_string().contains(&simplex_uuid.to_string())
                || err.to_string().contains("neighbor")
        );
    }

    #[test]
    fn test_tds_snapshot_rejects_wrong_neighbor_slot_count() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut snapshot = raw_snapshot_from_tds(dt.tds());
        let simplex_uuid = snapshot
            .simplices
            .first()
            .expect("snapshot should contain a simplex")
            .uuid;

        snapshot
            .simplex_neighbors
            .insert(simplex_uuid, vec![None, None]);

        let err = snapshot
            .parse()
            .expect_err("wrong neighbor slot count should be rejected");

        assert_matches!(
            err,
            TdsSnapshotError::InvalidSimplex {
                simplex_uuid: found,
                source: SimplexValidationError::InvalidNeighborsLength { .. },
            } if found == simplex_uuid
        );
    }

    #[test]
    fn test_tds_snapshot_rejects_dangling_runtime_neighbor_key() {
        let verts = initial_simplex_vertices_3d();
        let dt = DelaunayTriangulation::try_new(&verts).unwrap();
        let mut tds = dt.tds().clone();
        let simplex_key = tds
            .simplices()
            .next()
            .map(|(simplex_key, _simplex)| simplex_key)
            .expect("test TDS should contain a simplex");
        let simplex = tds
            .simplex_mut(simplex_key)
            .expect("test simplex key should still resolve");
        simplex
            .set_neighbors_from_keys([
                Some(SimplexKey::from(KeyData::from_ffi(0xBAD))),
                None,
                None,
                None,
            ])
            .expect("fixture neighbor arity should match");

        let err = TdsSnapshot::from_tds(&tds)
            .expect_err("snapshotting dangling runtime neighbor key should fail");

        assert_matches!(err, TdsSnapshotError::SourceValidationFailed { .. });
    }

    #[test]
    fn test_tds_snapshot_vertex_records_can_use_existing_vertex_type() {
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point");
        let vertex =
            Vertex::try_new_with_uuid(point, Uuid::new_v4(), Some(7_i32)).expect("valid vertex");

        let json = serde_json::to_value(vertex).expect("vertex should serialize");

        assert!(json.get("uuid").is_some());
        assert!(json.get("point").is_some());
        assert!(json.get("data").is_some());
        assert!(
            json.get("incident_simplex").is_none(),
            "runtime incident simplex keys must not enter snapshot vertex records"
        );
    }
}
