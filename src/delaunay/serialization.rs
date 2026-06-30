//! Serialization support for Delaunay triangulations.

#![forbid(unsafe_code)]

use crate::core::tds::Tds;
use crate::core::traits::data_type::DataType;
use crate::geometry::kernel::{Kernel, RobustKernel};
use crate::triangulation::DelaunayTriangulation;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<K, U, V, const D: usize> Serialize for DelaunayTriangulation<K, U, V, D>
where
    K: Kernel<D>,
    U: DataType,
    V: DataType,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.tri.tds.serialize(serializer)
    }
}

/// Custom `Deserialize` implementation for `RobustKernel<f64>` with no custom data.
///
/// Kernels are stateless and can be reconstructed on deserialization. This
/// implementation only serializes the `Tds`, which contains all geometric and
/// topological data, then reconstructs the kernel wrapper on deserialization.
///
/// # Note on Locate Hint Persistence
///
/// The internal `insertion_state.last_inserted_simplex` locate hint is not
/// serialized. Deserialization reconstructs a fresh triangulation via
/// [`DelaunayTriangulation::try_from_tds`], which resets the hint to `None`.
/// This only affects performance for the first few insertions after loading.
///
/// # Usage with Other Kernels
///
/// For other kernels such as `AdaptiveKernel` or `FastKernel`, or custom data
/// types, deserialize the `Tds` directly and reconstruct with
/// [`DelaunayTriangulation::try_from_tds`]:
///
/// ```rust
/// # use delaunay::prelude::geometry::*;
/// # use delaunay::prelude::tds::Tds;
/// # use delaunay::prelude::construction::{DelaunayTriangulation, DelaunayTriangulationBuilder};
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
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;
/// let json = serde_json::to_string(&dt)?;
///
/// let tds: Tds<(), (), 3> = serde_json::from_str(&json)?;
/// let dt_adaptive = DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new())?;
/// # let _ = dt_adaptive;
/// # Ok(())
/// # }
/// ```
impl<'de, const D: usize> Deserialize<'de> for DelaunayTriangulation<RobustKernel<f64>, (), (), D>
where
    Tds<(), (), D>: Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let tds = Tds::deserialize(deserializer)?;
        Self::try_from_tds(tds, RobustKernel::new()).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::simplex::Simplex;
    use crate::core::tds::TriangulationConstructionState;
    use crate::geometry::kernel::AdaptiveKernel;
    use crate::vertex;
    use std::sync::Once;

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
        tds.construction_state = TriangulationConstructionState::Constructed;
        tds.assign_neighbors().unwrap();
        tds.assign_incident_simplices().unwrap();
        tds
    }

    #[test]
    fn robust_deserialize_rejects_non_delaunay_connectivity() {
        init_tracing();
        let json = serde_json::to_string(&non_delaunay_quad_tds()).unwrap();

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
    fn serde_roundtrip_uses_custom_deserialize_impl() {
        init_tracing();
        let vertices = [
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::try_new(&vertices).unwrap();

        let json = serde_json::to_string(&dt).unwrap();

        let tds: Tds<(), (), 3> = serde_json::from_str(&json).unwrap();
        let roundtrip_adaptive =
            DelaunayTriangulation::try_from_tds(tds, AdaptiveKernel::new()).unwrap();

        assert_eq!(
            roundtrip_adaptive.number_of_vertices(),
            dt.number_of_vertices()
        );
        assert_eq!(
            roundtrip_adaptive.number_of_simplices(),
            dt.number_of_simplices()
        );
        assert!(
            roundtrip_adaptive
                .insertion_state
                .last_inserted_simplex
                .is_none()
        );

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
        assert!(
            roundtrip_robust
                .insertion_state
                .last_inserted_simplex
                .is_none()
        );
    }
}
