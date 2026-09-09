//! Exact UUID persistence for Levels 1–4 triangulations.
//!
//! The outer serde envelope embeds a CBOR TDS image so floating-point bits do
//! not depend on the outer codec. Loading validates Levels 1–4 strictly, with
//! no orientation normalization, connectivity changes, or Level 5 work.
//! Runtime keys, caches, kernel configuration, and construction provenance are
//! not portable; keys and incidence are rebuilt from UUIDs. Untrusted metadata
//! never supplies a high-dimensional PL-link proof.

use crate::core::tds::{RawTdsSnapshot, Tds, ValidatedTdsSerialization};
use crate::core::traits::data_type::{DataDeserialize, DataSerialize, DataType};
use crate::geometry::kernel::Kernel;
use crate::topology::traits::topological_space::{GlobalTopology, ToroidalConstructionMode};
use crate::triangulation::Triangulation;
use crate::triangulation::builder::{TriangulationBuildFailure, TriangulationBuilder};
use crate::triangulation::validation::{TopologyGuarantee, ValidationPolicy};

use ciborium::Value as CborValue;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de, ser};

/// Version of the exact Levels 1–4 persistence envelope.
pub const TRIANGULATION_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// Separates the payload codec representation from the entity's optional data slot.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredPayload<T> {
    value: T,
}

/// Rejects CBOR's ambiguous null representation before it can change a payload on reload.
fn contains_null(value: &CborValue) -> bool {
    match value {
        CborValue::Null => true,
        CborValue::Array(values) => values.iter().any(contains_null),
        CborValue::Map(entries) => entries
            .iter()
            .any(|(key, value)| contains_null(key) || contains_null(value)),
        CborValue::Tag(_, value) => contains_null(value),
        _ => false,
    }
}

/// Captures each borrowed payload once, so validation and encoding observe the same value.
fn capture_payload<T: Serialize, E: ser::Error>(
    payload: &T,
) -> Result<StoredPayload<CborValue>, E> {
    let value = CborValue::serialized(payload).map_err(E::custom)?;
    if contains_null(&value) {
        return Err(E::custom(
            "triangulation payload contains ambiguous CBOR null/unit data; use an explicitly tagged payload enum",
        ));
    }
    Ok(StoredPayload { value })
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredGuarantee {
    Pseudomanifold,
    PlManifold,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredPolicy {
    Never,
    ExplicitOnly,
    OnSuspicion,
    Always,
    DebugOnly,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StoredToroidalMode {
    PeriodicImagePoint,
    Explicit,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StoredTopology {
    Euclidean,
    Toroidal {
        period_bits: Vec<u64>,
        mode: StoredToroidalMode,
    },
    Spherical,
    Hyperbolic,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotWire {
    schema_version: u32,
    dimension: usize,
    tds: Vec<u8>,
    topology_guarantee: StoredGuarantee,
    global_topology: StoredTopology,
    validation_policy: StoredPolicy,
}

/// Decoded Levels 1–2 storage and context for strict Level 4 restoration.
///
/// Deserialize this type to separate codec errors from the typed, recoverable
/// [`TriangulationBuildFailure`] returned by [`Self::try_into_triangulation`].
/// Decoding validates UUID relationships and TDS invariants; it does not yet
/// certify topology or realization. The kernel is chosen explicitly at restoration.
/// Serialization through [`Triangulation`] borrows payloads and requires only
/// `Serialize`, with no `Clone` bound. Restoring the owner retains the usual
/// [`DataType`] bounds of [`TriangulationBuilder`].
/// Present payloads containing null/unit values are rejected during serialization
/// because CBOR cannot distinguish all nested `Option` states. Use explicit
/// tagged enums for those values; an absent entity payload remains supported.
#[derive(Debug)]
pub struct TriangulationSnapshot<U, V, const D: usize> {
    tds: Tds<U, V, D>,
    topology_guarantee: TopologyGuarantee,
    global_topology: GlobalTopology<D>,
    validation_policy: ValidationPolicy,
}

impl<U, V, const D: usize> TriangulationSnapshot<U, V, D>
where
    U: DataType,
    V: DataType,
{
    /// Restores exact connectivity with the supplied kernel and proves Levels 1–4.
    ///
    /// UUIDs, simplex vertex order, neighbor slots, periodic lift offsets,
    /// topology context, policy, and payloads are retained. Runtime keys are
    /// newly allocated and should be resolved by UUID. No Level 5 work occurs.
    ///
    /// # Errors
    ///
    /// Returns the unchanged decoded TDS with a typed [`TriangulationBuildFailure`]
    /// when strict certification fails. Nontrivial high-dimensional PL manifolds
    /// whose link proof depends on construction history may be rejected: serialized
    /// metadata is not accepted as proof of that history.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};
    /// use delaunay::prelude::triangulation::{RobustKernel, TriangulationSnapshot};
    /// # #[derive(Debug, thiserror::Error)]
    /// # enum ExampleError {
    /// #   #[error(transparent)] Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
    /// #   #[error(transparent)] Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
    /// #   #[error(transparent)] Codec(#[from] serde_json::Error),
    /// #   #[error(transparent)] Restore(#[from] delaunay::TriangulationBuildFailure<(), (), 2>),
    /// #   #[error(transparent)] Realization(#[from] delaunay::TriangulationRealizationValidationError),
    /// # }
    /// # fn main() -> Result<(), ExampleError> {
    /// let vertices = [vertex![0.0, 0.0]?, vertex![1.0, 0.0]?, vertex![0.0, 1.0]?];
    /// let tri = DelaunayTriangulationBuilder::new(&vertices).build_triangulation()?;
    /// let json = serde_json::to_string(&tri)?;
    /// let snapshot: TriangulationSnapshot<(), (), 2> = serde_json::from_str(&json)?;
    /// let restored = snapshot.try_into_triangulation(RobustKernel::new())?;
    /// restored.validate_realization()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_into_triangulation<K>(
        self,
        kernel: K,
    ) -> Result<Triangulation<K, U, V, D>, TriangulationBuildFailure<U, V, D>>
    where
        K: Kernel<D, Scalar = f64>,
    {
        TriangulationBuilder::new(self.tds, kernel)
            .topology_guarantee(self.topology_guarantee)
            .global_topology(self.global_topology)
            .validation_policy(self.validation_policy)
            .build()
    }
}

impl<K, U, V, const D: usize> Serialize for Triangulation<K, U, V, D>
where
    U: DataSerialize,
    V: DataSerialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let validated =
            ValidatedTdsSerialization::try_new(&self.tds).map_err(ser::Error::custom)?;
        let snapshot = validated.snapshot().map_err(ser::Error::custom)?;
        let wrapped = snapshot.try_map_payloads(
            capture_payload::<_, S::Error>,
            capture_payload::<_, S::Error>,
        )?;
        let mut tds = Vec::new();
        ciborium::ser::into_writer(&wrapped.into_raw(), &mut tds).map_err(ser::Error::custom)?;
        SnapshotWire {
            schema_version: TRIANGULATION_SNAPSHOT_SCHEMA_VERSION,
            dimension: D,
            tds,
            topology_guarantee: match self.topology_guarantee {
                TopologyGuarantee::Pseudomanifold => StoredGuarantee::Pseudomanifold,
                TopologyGuarantee::PLManifold => StoredGuarantee::PlManifold,
            },
            global_topology: match self.global_topology {
                GlobalTopology::Euclidean => StoredTopology::Euclidean,
                GlobalTopology::Toroidal { domain, mode } => StoredTopology::Toroidal {
                    period_bits: domain
                        .periods()
                        .iter()
                        .map(|period| period.to_bits())
                        .collect(),
                    mode: match mode {
                        ToroidalConstructionMode::PeriodicImagePoint => {
                            StoredToroidalMode::PeriodicImagePoint
                        }
                        ToroidalConstructionMode::Explicit => StoredToroidalMode::Explicit,
                    },
                },
                GlobalTopology::Spherical => StoredTopology::Spherical,
                GlobalTopology::Hyperbolic => StoredTopology::Hyperbolic,
            },
            validation_policy: match self.validation_policy {
                ValidationPolicy::Never => StoredPolicy::Never,
                ValidationPolicy::ExplicitOnly => StoredPolicy::ExplicitOnly,
                ValidationPolicy::OnSuspicion => StoredPolicy::OnSuspicion,
                ValidationPolicy::Always => StoredPolicy::Always,
                ValidationPolicy::DebugOnly => StoredPolicy::DebugOnly,
            },
        }
        .serialize(serializer)
    }
}

impl<'de, U, V, const D: usize> Deserialize<'de> for TriangulationSnapshot<U, V, D>
where
    U: DataDeserialize,
    V: DataDeserialize,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let wire = SnapshotWire::deserialize(deserializer)?;
        if wire.schema_version != TRIANGULATION_SNAPSHOT_SCHEMA_VERSION {
            return Err(de::Error::custom(format_args!(
                "unsupported triangulation snapshot version {}",
                wire.schema_version
            )));
        }
        if wire.dimension != D {
            return Err(de::Error::custom(format_args!(
                "snapshot dimension {} does not match {D}",
                wire.dimension
            )));
        }
        let global_topology = match wire.global_topology {
            StoredTopology::Euclidean => GlobalTopology::Euclidean,
            StoredTopology::Toroidal { period_bits, mode } => {
                let periods: [u64; D] = period_bits.try_into().map_err(|bits: Vec<u64>| {
                    de::Error::custom(format_args!("expected {D} periods, got {}", bits.len()))
                })?;
                GlobalTopology::try_toroidal(
                    periods.map(f64::from_bits),
                    match mode {
                        StoredToroidalMode::PeriodicImagePoint => {
                            ToroidalConstructionMode::PeriodicImagePoint
                        }
                        StoredToroidalMode::Explicit => ToroidalConstructionMode::Explicit,
                    },
                )
                .map_err(de::Error::custom)?
            }
            StoredTopology::Spherical => GlobalTopology::Spherical,
            StoredTopology::Hyperbolic => GlobalTopology::Hyperbolic,
        };
        let mut remaining = wire.tds.as_slice();
        let raw: RawTdsSnapshot<StoredPayload<U>, StoredPayload<V>, D> =
            ciborium::de::from_reader(&mut remaining).map_err(de::Error::custom)?;
        if !remaining.is_empty() {
            return Err(de::Error::custom(
                "trailing bytes in triangulation TDS image",
            ));
        }
        let tds = raw
            .parse()
            .map_err(de::Error::custom)?
            .map_payloads(|payload| payload.value, |payload| payload.value)
            .into_tds()
            .map_err(de::Error::custom)?;
        Ok(Self {
            tds,
            topology_guarantee: match wire.topology_guarantee {
                StoredGuarantee::Pseudomanifold => TopologyGuarantee::Pseudomanifold,
                StoredGuarantee::PlManifold => TopologyGuarantee::PLManifold,
            },
            global_topology,
            validation_policy: match wire.validation_policy {
                StoredPolicy::Never => ValidationPolicy::Never,
                StoredPolicy::ExplicitOnly => ValidationPolicy::ExplicitOnly,
                StoredPolicy::OnSuspicion => ValidationPolicy::OnSuspicion,
                StoredPolicy::Always => ValidationPolicy::Always,
                StoredPolicy::DebugOnly => ValidationPolicy::DebugOnly,
            },
        })
    }
}

impl<'de, K, U, V, const D: usize> Deserialize<'de> for Triangulation<K, U, V, D>
where
    K: Kernel<D, Scalar = f64> + Default,
    U: DataType,
    V: DataType,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        TriangulationSnapshot::deserialize(deserializer)?
            .try_into_triangulation(K::default())
            .map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tds::TdsBuilder;
    use crate::geometry::kernel::RobustKernel;
    use crate::triangulation::builder::TriangulationBuilderError;
    use crate::triangulation::realization::TriangulationRealizationValidationError;
    use crate::triangulation::validation::{
        TopologyConstructionProvenance, ValidationConfigurationError,
    };
    use crate::{DelaunayTriangulationBuilder, vertex};
    use serde_json::Value;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    struct NonClonePayload(u32);

    /// Builds one recognizable PL ball with payloads and nontrivial coordinate bits.
    fn sample<const D: usize>() -> Triangulation<RobustKernel<f64>, u32, u32, D> {
        let mut vertices = vec![vertex!([0.0; D]; data = 7_u32).unwrap()];
        for axis in 0..D {
            let mut coordinates = [0.0; D];
            coordinates[axis] = f64::from_bits(1.0_f64.to_bits() + 1);
            vertices.push(vertex!(coordinates; data = 9_u32).unwrap());
        }
        let simplices = [(0..=D).collect::<Vec<_>>()];
        let tds = TdsBuilder::new(&vertices, &simplices)
            .simplex_data_type::<u32>()
            .build()
            .unwrap();
        let mut tri = TriangulationBuilder::new(tds, RobustKernel::new())
            .validation_policy(ValidationPolicy::Always)
            .canonicalizing()
            .build()
            .unwrap();
        let key = tri.simplices().next().unwrap().0;
        tri.set_simplex_data(key, Some(42)).unwrap();
        tri
    }

    /// Checks bitwise coordinates, UUID relations, policy, and optional payload presence.
    fn round_trip<const D: usize>() {
        let tri = sample::<D>();
        let json = serde_json::to_string(&tri).unwrap();
        let snapshot: TriangulationSnapshot<u32, u32, D> = serde_json::from_str(&json).unwrap();
        let restored = snapshot
            .try_into_triangulation(RobustKernel::new())
            .unwrap();
        assert_eq!(restored.validation_policy(), ValidationPolicy::Always);
        assert_eq!(restored.global_topology(), tri.global_topology());
        assert_eq!(restored.topology_guarantee(), tri.topology_guarantee());
        for (_, vertex) in tri.vertices() {
            let key = restored.vertex_key_from_uuid(&vertex.uuid()).unwrap();
            let actual = restored.vertex(key).unwrap();
            assert_eq!(
                actual.point().coords().map(f64::to_bits),
                vertex.point().coords().map(f64::to_bits)
            );
            assert_eq!(actual.data(), vertex.data());
        }
        assert_eq!(restored.simplices().next().unwrap().1.data(), Some(&42));
        assert_eq!(
            serde_json::to_value(&restored.tds).unwrap(),
            serde_json::to_value(&tri.tds).unwrap()
        );
        restored.validate_realization().unwrap();
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&tri, &mut bytes).unwrap();
        let from_cbor: Triangulation<RobustKernel<f64>, u32, u32, D> =
            ciborium::de::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(
            serde_json::to_value(from_cbor.tds).unwrap(),
            serde_json::to_value(tri.tds).unwrap()
        );
    }

    /// Builds a realized quotient for detecting an incompatible stored chart mode.
    fn periodic_torus() -> Triangulation<RobustKernel<f64>, u32, (), 2> {
        let vertices = (0_u32..7)
            .map(|index| {
                let index_f64 = f64::from(index);
                vertex!([
                    0.9_f64.mul_add(((index_f64 + 1.0) * 0.618_033_988_749_894_8).fract(), 0.05),
                    0.9_f64.mul_add(((index_f64 + 1.0) * 0.414_213_562_373_095_03).fract(), 0.05),
                ]; data = index)
                .unwrap()
            })
            .collect::<Vec<_>>();
        DelaunayTriangulationBuilder::new(&vertices)
            .try_toroidal([1.0; 2])
            .unwrap()
            .build_triangulation_with_kernel(&RobustKernel::new())
            .unwrap()
    }

    macro_rules! dimension_tests {
        ($($dimension:literal),+) => { $(pastey::paste! {
            #[test]
            fn [<exact_snapshot_round_trip_ $dimension d>]() { round_trip::<$dimension>(); }
        })+ };
    }
    dimension_tests!(2, 3, 4, 5);

    #[test]
    fn nested_null_payloads_are_rejected_before_encoding() {
        let payloads = [
            CborValue::Array(vec![CborValue::from(7), CborValue::Null]),
            CborValue::Map(vec![(CborValue::from("value"), CborValue::Null)]),
            CborValue::Map(vec![(CborValue::Null, CborValue::from(7))]),
            CborValue::Tag(42, Box::new(CborValue::Null)),
        ];
        for payload in payloads {
            let result = capture_payload::<_, serde_json::Error>(&payload);
            let error = result
                .err()
                .expect("nested null must not be silently collapsed");
            assert!(error.to_string().contains("ambiguous CBOR null/unit data"));
        }
    }

    #[test]
    fn structured_payload_capture_retains_arrays_maps_and_tags() {
        let payload = CborValue::Map(vec![(
            CborValue::from("samples"),
            CborValue::Array(vec![
                CborValue::from(7),
                CborValue::Tag(42, Box::new(CborValue::from(9))),
            ]),
        )]);
        let captured = capture_payload::<_, serde_json::Error>(&payload).unwrap();
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&captured, &mut bytes).unwrap();
        let restored: StoredPayload<CborValue> =
            ciborium::de::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(restored.value, payload);
    }

    #[test]
    fn snapshots_preserve_each_compatible_guarantee_and_validation_policy() {
        let storage = sample::<2>().into_tds();
        let expected = serde_json::to_value(&storage).unwrap();
        for guarantee in [
            TopologyGuarantee::PLManifold,
            TopologyGuarantee::Pseudomanifold,
        ] {
            for policy in [
                ValidationPolicy::Never,
                ValidationPolicy::ExplicitOnly,
                ValidationPolicy::OnSuspicion,
                ValidationPolicy::Always,
                ValidationPolicy::DebugOnly,
            ] {
                // Never is the one policy explicitly forbidden for PL manifolds.
                if guarantee == TopologyGuarantee::PLManifold && policy == ValidationPolicy::Never {
                    continue;
                }
                let tri = TriangulationBuilder::new(storage.clone(), RobustKernel::new())
                    .topology_guarantee(guarantee)
                    .validation_policy(policy)
                    .build()
                    .unwrap();
                let restored: Triangulation<RobustKernel<f64>, u32, u32, 2> =
                    serde_json::from_str(&serde_json::to_string(&tri).unwrap()).unwrap();
                assert_eq!(restored.topology_guarantee(), guarantee);
                assert_eq!(restored.validation_policy(), policy);
                assert_eq!(serde_json::to_value(&restored.tds).unwrap(), expected);
                restored.validate_realization().unwrap();
            }
        }
    }

    #[test]
    fn explicit_mode_cannot_be_reinterpreted_as_periodic_image_geometry() {
        let tri = periodic_torus();
        tri.validate_realization().unwrap();
        let expected = serde_json::to_value(&tri.tds).unwrap();
        assert!(tri.simplices().any(|(_, simplex)| {
            simplex
                .periodic_vertex_offsets()
                .is_some_and(|offsets| offsets.iter().flatten().any(|&offset| offset != 0))
        }));
        let mut encoded = serde_json::to_value(&tri).unwrap();
        encoded["global_topology"]["mode"] = Value::from("explicit");
        assert_eq!(
            encoded["global_topology"]["period_bits"],
            serde_json::to_value([1.0_f64.to_bits(); 2]).unwrap()
        );
        let decoded: TriangulationSnapshot<u32, (), 2> = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            decoded.global_topology,
            GlobalTopology::try_toroidal([1.0; 2], ToroidalConstructionMode::Explicit).unwrap()
        );
        assert_eq!(serde_json::to_value(&decoded.tds).unwrap(), expected);
        let failure = decoded
            .try_into_triangulation(RobustKernel::new())
            .unwrap_err();
        assert!(
            matches!(failure.reason(), TriangulationBuilderError::RealizationValidation { source }
            if matches!(source.as_ref(), TriangulationRealizationValidationError::NegativeSimplexOrientation { .. }))
        );
        assert_eq!(serde_json::to_value(failure.owner()).unwrap(), expected);
    }

    #[test]
    fn malformed_toroidal_domains_fail_during_snapshot_decoding() {
        let original = serde_json::to_value(sample::<2>()).unwrap();
        let unit = 1.0_f64.to_bits();
        for (periods, diagnostic) in [
            (vec![unit], "expected 2 periods, got 1"),
            (vec![unit; 3], "expected 2 periods, got 3"),
            (vec![unit, 0.0_f64.to_bits()], "axis 1"),
            (vec![unit, (-1.0_f64).to_bits()], "axis 1"),
            (vec![unit, f64::INFINITY.to_bits()], "axis 1"),
            (vec![unit, f64::NAN.to_bits()], "axis 1"),
        ] {
            let mut invalid = original.clone();
            invalid["global_topology"] = serde_json::json!({
                "kind": "toroidal", "mode": "explicit", "period_bits": periods,
            });
            let error =
                serde_json::from_value::<TriangulationSnapshot<u32, u32, 2>>(invalid).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn restored_policy_cannot_bypass_pl_manifold_requirements() {
        let mut invalid = serde_json::to_value(sample::<2>()).unwrap();
        invalid["validation_policy"] = Value::from("never");
        let decoded: TriangulationSnapshot<u32, u32, 2> = serde_json::from_value(invalid).unwrap();
        let expected = serde_json::to_value(&decoded.tds).unwrap();
        let failure = decoded
            .try_into_triangulation(RobustKernel::new())
            .unwrap_err();
        assert_eq!(
            failure.reason(),
            &TriangulationBuilderError::ValidationConfiguration {
                source: ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                    topology_guarantee: TopologyGuarantee::PLManifold,
                    validation_policy: ValidationPolicy::Never,
                },
            }
        );
        assert_eq!(serde_json::to_value(failure.owner()).unwrap(), expected);
    }

    #[test]
    fn decoded_curved_metadata_cannot_publish_an_unsupported_owner() {
        let empty = TriangulationBuilder::new(Tds::<(), (), 2>::empty(), RobustKernel::new())
            .build()
            .unwrap();
        let original = serde_json::to_value(empty).unwrap();
        for (topology, selected) in [
            ("spherical", GlobalTopology::Spherical),
            ("hyperbolic", GlobalTopology::Hyperbolic),
        ] {
            let mut invalid = original.clone();
            invalid["global_topology"] = serde_json::json!({"kind": topology});
            let decoded: TriangulationSnapshot<(), (), 2> =
                serde_json::from_value(invalid).unwrap();
            let expected = serde_json::to_value(&decoded.tds).unwrap();
            assert_eq!(decoded.global_topology, selected);
            let failure = decoded
                .try_into_triangulation(RobustKernel::new())
                .unwrap_err();
            assert!(
                matches!(failure.reason(), TriangulationBuilderError::RealizationValidation { source }
                if matches!(source.as_ref(), TriangulationRealizationValidationError::UnsupportedTopology {
                    topology, dimension: 2,
                } if *topology == selected.kind()))
            );
            assert_eq!(serde_json::to_value(failure.owner()).unwrap(), expected);
            failure.owner().validate().unwrap();
        }
    }

    #[test]
    fn serialization_and_transport_decoding_do_not_require_clone() {
        let vertices = [
            vertex!([0.0, 0.0]; data = 7_u32).unwrap(),
            vertex!([1.0, 0.0]; data = 8).unwrap(),
            vertex!([0.0, 1.0]; data = 9).unwrap(),
        ];
        let tds = TdsBuilder::new(&vertices, &[vec![0, 1, 2]])
            .simplex_data_type::<u32>()
            .build()
            .unwrap();
        let tri = TriangulationBuilder::new(tds, RobustKernel::new())
            .build()
            .unwrap();
        let snapshot: TriangulationSnapshot<NonClonePayload, NonClonePayload, 2> =
            serde_json::from_str(&serde_json::to_string(&tri).unwrap()).unwrap();
        // The fixture's geometry is already certified. Only the transparent
        // payload wrapper changes; no geometry or topology data is modified.
        let non_clone = Triangulation {
            kernel: RobustKernel::<f64>::new(),
            tds: snapshot.tds,
            topology_guarantee: snapshot.topology_guarantee,
            global_topology: snapshot.global_topology,
            validation_policy: snapshot.validation_policy,
            topology_construction_provenance: TopologyConstructionProvenance::Unproven,
        };
        let encoded = serde_json::to_string(&non_clone).unwrap();
        let decoded: TriangulationSnapshot<NonClonePayload, NonClonePayload, 2> =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            decoded
                .tds
                .vertices()
                .map(|(_, vertex)| vertex.data().unwrap().0)
                .collect::<Vec<_>>(),
            [7, 8, 9]
        );
        assert_eq!(non_clone.to_visualization_data().unwrap().vertices.len(), 3);
    }

    #[test]
    fn malformed_envelopes_and_trailing_images_are_rejected() {
        let original = serde_json::to_value(sample::<2>()).unwrap();
        for (field, value) in [
            ("schema_version", Value::from(2)),
            ("dimension", Value::from(3)),
        ] {
            let mut invalid = original.clone();
            invalid[field] = value;
            assert!(serde_json::from_value::<TriangulationSnapshot<u32, u32, 2>>(invalid).is_err());
        }
        let mut trailing = original.clone();
        trailing["tds"].as_array_mut().unwrap().push(Value::from(0));
        assert!(serde_json::from_value::<TriangulationSnapshot<u32, u32, 2>>(trailing).is_err());
        let mut forged = original;
        forged["construction_provenance"] = Value::from("euclidean_delaunay_insertion");
        assert!(serde_json::from_value::<TriangulationSnapshot<u32, u32, 2>>(forged).is_err());
    }

    #[test]
    fn ambiguous_present_payloads_are_rejected_without_cloning_or_changing_state() {
        for payload in [None, Some(())] {
            let vertices = [
                vertex!([0.0, 0.0]; data = payload).unwrap(),
                vertex!([1.0, 0.0]; data = payload).unwrap(),
                vertex!([0.0, 1.0]; data = payload).unwrap(),
            ];
            let tds = TdsBuilder::new(&vertices, &[vec![0, 1, 2]])
                .build()
                .unwrap();
            let tri = TriangulationBuilder::new(tds, RobustKernel::new())
                .build()
                .unwrap();
            let error = serde_json::to_string(&tri).unwrap_err();
            assert!(error.to_string().contains("ambiguous CBOR null/unit data"));
            assert!(
                tri.vertices()
                    .all(|(_, vertex)| vertex.data() == Some(&payload))
            );
        }
    }

    #[test]
    fn restored_high_dimensional_proof_is_not_inferred_from_metadata() {
        let mut tri = sample::<4>();
        tri.insert_vertex(vertex!([0.1; 4]; data = 7_u32).unwrap())
            .unwrap();
        let snapshot: TriangulationSnapshot<u32, u32, 4> =
            serde_json::from_str(&serde_json::to_string(&tri).unwrap()).unwrap();
        let expected = serde_json::to_value(&snapshot.tds).unwrap();
        let failure = snapshot
            .try_into_triangulation(RobustKernel::new())
            .unwrap_err();
        assert!(matches!(
            failure.reason(),
            TriangulationBuilderError::TopologyValidation { .. }
        ));
        assert_eq!(serde_json::to_value(failure.owner()).unwrap(), expected);
    }
}
