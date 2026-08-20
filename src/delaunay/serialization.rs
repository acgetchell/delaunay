//! Serialization support for Delaunay triangulations.

#![forbid(unsafe_code)]

use crate::core::tds::Tds;
use crate::core::traits::data_type::{DataSerialize, DataType};
use crate::delaunay_model::DelaunayTriangulation;
use crate::geometry::kernel::RobustKernel;
use crate::topology::traits::topological_space::{
    GlobalTopology, ToroidalConstructionMode, ToroidalDomainError,
};
use crate::triangulation::validation::{
    TopologyGuarantee, ValidationConfigurationError, ValidationPolicy,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

const DELAUNAY_SERIALIZATION_SCHEMA_VERSION: u32 = 1;

/// Owner-level persistence DTO that keeps proof context beside canonical TDS storage.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DelaunayTriangulationWire<T> {
    schema_version: u32,
    tds: T,
    topology_guarantee: TopologyGuaranteeWire,
    global_topology: GlobalTopologyWire,
    validation_policy: ValidationPolicyWire,
}

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
enum GlobalTopologyWire {
    Euclidean {},
    Toroidal {
        periods: Vec<f64>,
        mode: ToroidalConstructionModeWire,
    },
    Spherical {},
    Hyperbolic {},
}

impl<const D: usize> From<GlobalTopology<D>> for GlobalTopologyWire {
    fn from(value: GlobalTopology<D>) -> Self {
        match value {
            GlobalTopology::Euclidean => Self::Euclidean {},
            GlobalTopology::Toroidal { domain, mode } => Self::Toroidal {
                periods: domain.periods().to_vec(),
                mode: mode.into(),
            },
            GlobalTopology::Spherical => Self::Spherical {},
            GlobalTopology::Hyperbolic => Self::Hyperbolic {},
        }
    }
}

impl GlobalTopologyWire {
    /// Parses raw topology metadata through the domain constructor before proof restoration.
    fn try_into_global_topology<const D: usize>(
        self,
    ) -> Result<GlobalTopology<D>, DelaunayTriangulationWireError> {
        match self {
            Self::Euclidean {} => Ok(GlobalTopology::Euclidean),
            Self::Toroidal { periods, mode } => {
                let actual = periods.len();
                let periods = periods.try_into().map_err(|_| {
                    DelaunayTriangulationWireError::ToroidalDimensionMismatch {
                        expected: D,
                        actual,
                    }
                })?;
                GlobalTopology::try_toroidal(periods, mode.into())
                    .map_err(|source| DelaunayTriangulationWireError::ToroidalDomain { source })
            }
            Self::Spherical {} => Ok(GlobalTopology::Spherical),
            Self::Hyperbolic {} => Ok(GlobalTopology::Hyperbolic),
        }
    }
}

/// Typed wire-level failures detected before proof-bearing reconstruction starts.
#[derive(Clone, Debug, Error, PartialEq)]
enum DelaunayTriangulationWireError {
    #[error(
        "unsupported Delaunay triangulation serialization schema version {actual}; expected {expected}"
    )]
    UnsupportedSchemaVersion { actual: u32, expected: u32 },
    #[error("toroidal topology has {actual} periods; expected {expected}")]
    ToroidalDimensionMismatch { expected: usize, actual: usize },
    #[error("invalid serialized toroidal domain: {source}")]
    ToroidalDomain {
        #[source]
        source: ToroidalDomainError,
    },
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
        DelaunayTriangulationWire {
            schema_version: DELAUNAY_SERIALIZATION_SCHEMA_VERSION,
            tds: &self.tri.tds,
            topology_guarantee: self.topology_guarantee().into(),
            global_topology: self.global_topology().into(),
            validation_policy: self.validation_policy().into(),
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
/// [`DelaunayTriangulation::try_from_tds_with_topology_context`], which resets
/// the hint to `None`.
/// This only affects performance for the first few insertions after loading.
///
/// # Usage with Other Kernels
///
/// The direct serde boundary reconstructs [`RobustKernel<f64>`] because a
/// caller-supplied kernel cannot be obtained from `Deserialize`. Workflows
/// requiring another kernel should persist their [`Tds`] and owner-level
/// topology context explicitly, then reconstruct with
/// [`DelaunayTriangulation::try_from_tds_with_topology_context`].
///
/// ```rust
/// # use delaunay::prelude::geometry::AdaptiveKernel;
/// # use delaunay::prelude::tds::Tds;
/// # use delaunay::prelude::construction::{
/// #     DelaunayTriangulation, DelaunayTriangulationBuilder, GlobalTopology,
/// #     TopologyGuarantee,
/// # };
/// # use delaunay::prelude::validation::DelaunayTdsRefinementError;
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Serde(#[from] serde_json::Error),
/// #     #[error(transparent)]
/// #     Validation(#[from] delaunay::DelaunayTriangulationValidationError),
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
/// let topology_guarantee = dt.topology_guarantee();
/// let global_topology = dt.global_topology();
/// let tds_json = serde_json::to_string(&dt.into_triangulation().into_tds())?;
///
/// let tds: Tds<(), (), 3> = serde_json::from_str(&tds_json)?;
/// let dt_adaptive = DelaunayTriangulation::try_from_tds_with_topology_context(
///     tds,
///     AdaptiveKernel::new(),
///     topology_guarantee,
///     global_topology,
/// )
/// .map_err(DelaunayTdsRefinementError::into_reason)?;
/// # let _ = dt_adaptive;
/// # Ok(())
/// # }
/// ```
impl<'de, U, V, const D: usize> Deserialize<'de>
    for DelaunayTriangulation<RobustKernel<f64>, U, V, D>
where
    U: DataType,
    V: DataType,
    Tds<U, V, D>: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let wire = DelaunayTriangulationWire::<Tds<U, V, D>>::deserialize(deserializer)?;
        if wire.schema_version != DELAUNAY_SERIALIZATION_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                DelaunayTriangulationWireError::UnsupportedSchemaVersion {
                    actual: wire.schema_version,
                    expected: DELAUNAY_SERIALIZATION_SCHEMA_VERSION,
                },
            ));
        }

        let topology_guarantee: TopologyGuarantee = wire.topology_guarantee.into();
        let validation_policy: ValidationPolicy = wire.validation_policy.into();
        if !topology_guarantee.is_compatible_with_policy(validation_policy) {
            return Err(serde::de::Error::custom(
                ValidationConfigurationError::IncompatibleTopologyAndValidationPolicy {
                    topology_guarantee,
                    validation_policy,
                },
            ));
        }
        let global_topology = wire
            .global_topology
            .try_into_global_topology::<D>()
            .map_err(serde::de::Error::custom)?;

        let mut triangulation = Self::try_from_tds_with_topology_context(
            wire.tds,
            RobustKernel::new(),
            topology_guarantee,
            global_topology,
        )
        .map_err(serde::de::Error::custom)?;
        triangulation
            .try_set_validation_policy(validation_policy)
            .map_err(serde::de::Error::custom)?;
        Ok(triangulation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::operations::DelaunayInsertionState;
    use crate::core::simplex::Simplex;
    use crate::delaunay_model::EuclideanDelaunayReportDomain;
    use crate::topology::traits::topological_space::GlobalTopology;
    use crate::triangulation::Triangulation;
    use crate::triangulation::validation::{TopologyGuarantee, ValidationPolicy};
    use crate::vertex;
    use std::sync::Once;

    struct NotAKernel;

    #[derive(Serialize)]
    struct SerializeOnlyPayload(String);

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

    #[test]
    fn global_topology_wire_accepts_unit_variants_without_payload_fields() {
        let euclidean: GlobalTopologyWire =
            serde_json::from_value(serde_json::json!({ "kind": "euclidean" })).unwrap();
        let spherical: GlobalTopologyWire =
            serde_json::from_value(serde_json::json!({ "kind": "spherical" })).unwrap();
        let hyperbolic: GlobalTopologyWire =
            serde_json::from_value(serde_json::json!({ "kind": "hyperbolic" })).unwrap();

        assert!(matches!(euclidean, GlobalTopologyWire::Euclidean {}));
        assert!(matches!(spherical, GlobalTopologyWire::Spherical {}));
        assert!(matches!(hyperbolic, GlobalTopologyWire::Hyperbolic {}));
    }

    #[test]
    fn global_topology_wire_rejects_unknown_fields_for_unit_variants() {
        for kind in ["euclidean", "spherical", "hyperbolic"] {
            let error = serde_json::from_value::<GlobalTopologyWire>(serde_json::json!({
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
        let error = serde_json::from_value::<GlobalTopologyWire>(serde_json::json!({
            "kind": "toroidal",
            "periods": [1.0, 1.0],
            "mode": "explicit",
            "unexpected": true
        }))
        .expect_err("toroidal topology must reject fields outside its wire schema");

        assert!(error.to_string().contains("unknown field `unexpected`"));
    }

    #[test]
    fn robust_deserialize_rejects_non_delaunay_connectivity() {
        init_tracing();
        let json = serde_json::to_string(&DelaunayTriangulationWire {
            schema_version: DELAUNAY_SERIALIZATION_SCHEMA_VERSION,
            tds: non_delaunay_quad_tds(),
            topology_guarantee: TopologyGuaranteeWire::PlManifold,
            global_topology: GlobalTopologyWire::Euclidean {},
            validation_policy: ValidationPolicyWire::ExplicitOnly,
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
        checkpoint["schema_version"] = serde_json::json!(2);

        let error = serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
            checkpoint,
        )
        .expect_err("unknown checkpoint versions must fail explicitly");

        assert!(
            error
                .to_string()
                .contains("unsupported Delaunay triangulation serialization schema version 2")
        );
    }

    #[test]
    fn robust_deserialize_parses_toroidal_metadata_before_proof_restoration() {
        let dt: DelaunayTriangulation<_, (), (), 2> = DelaunayTriangulation::empty();
        let mut checkpoint = serde_json::to_value(dt).unwrap();
        checkpoint["global_topology"] = serde_json::json!({
            "kind": "toroidal",
            "periods": [1.0],
            "mode": "explicit"
        });

        let error = serde_json::from_value::<DelaunayTriangulation<RobustKernel<f64>, (), (), 2>>(
            checkpoint,
        )
        .expect_err("dimension-mismatched raw toroidal metadata must be rejected");

        assert!(error.to_string().contains("has 1 periods; expected 2"));
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
        assert_eq!(checkpoint["schema_version"], 1);
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
}
