//! Public-API coverage for versioned scientific checkpoint manifests.

#![forbid(unsafe_code)]

use delaunay::prelude::checkpoint::{
    DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM, DELAUNAY_CHECKPOINT_DIGEST_VERSION,
    DELAUNAY_CHECKPOINT_MANIFEST_VERSION, DELAUNAY_CHECKPOINT_SCHEMA_VERSION,
    DelaunayCheckpointError, DelaunayCheckpointManifest,
};
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, vertex,
};
use delaunay::prelude::geometry::RobustKernel;
use serde_json::{from_value, to_value};

macro_rules! checkpoint_manifest_case {
    ($dimension:literal, $wrong_dimension:literal, $vertices:expr, $f_vector:expr) => {
        pastey::paste! {
            #[test]
            fn [<downstream_checkpoint_manifest_is_inspectable_and_verified_on_load_ $dimension d>]() {
                let vertices = $vertices;
                let triangulation: DelaunayTriangulation<
                    RobustKernel<f64>,
                    (),
                    (),
                    $dimension,
                > = DelaunayTriangulationBuilder::new(&vertices)
                    .build_with_kernel(&RobustKernel::new())
                    .unwrap();

                let manifest: DelaunayCheckpointManifest =
                    triangulation.checkpoint_manifest().unwrap();
                assert_eq!(DELAUNAY_CHECKPOINT_SCHEMA_VERSION, 2);
                assert_eq!(
                    manifest.manifest_version,
                    DELAUNAY_CHECKPOINT_MANIFEST_VERSION
                );
                assert_eq!(manifest.dimension, $dimension);
                assert_eq!(manifest.f_vector, $f_vector);
                assert_eq!(manifest.euler_characteristic, 1);
                assert_eq!(manifest.digest.version, DELAUNAY_CHECKPOINT_DIGEST_VERSION);
                assert_eq!(
                    manifest.digest.algorithm.as_str(),
                    DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM
                );
                triangulation.verify_checkpoint_manifest(&manifest).unwrap();

                let mut wrong_dimension = manifest.clone();
                wrong_dimension.dimension = $wrong_dimension;
                assert_eq!(
                    triangulation.verify_checkpoint_manifest(&wrong_dimension),
                    Err(DelaunayCheckpointError::DimensionMismatch {
                        expected: $dimension,
                        actual: $wrong_dimension,
                    })
                );

                let checkpoint = to_value(triangulation).unwrap();
                assert_eq!(checkpoint["manifest"], to_value(&manifest).unwrap());
                let restored: DelaunayTriangulation<
                    RobustKernel<f64>,
                    (),
                    (),
                    $dimension,
                > = from_value(checkpoint).unwrap();
                assert_eq!(restored.checkpoint_manifest().unwrap(), manifest);
                restored.validate().unwrap();
            }
        }
    };
}

checkpoint_manifest_case!(
    2,
    3,
    [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ],
    vec![3, 3, 1]
);
checkpoint_manifest_case!(
    3,
    4,
    [
        vertex!([0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0]).unwrap(),
    ],
    vec![4, 6, 4, 1]
);
checkpoint_manifest_case!(
    4,
    5,
    [
        vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
    ],
    vec![5, 10, 10, 5, 1]
);
checkpoint_manifest_case!(
    5,
    4,
    [
        vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
        vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
    ],
    vec![6, 15, 20, 15, 6, 1]
);
