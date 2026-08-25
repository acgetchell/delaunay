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

#[test]
fn downstream_checkpoint_manifest_is_inspectable_and_verified_on_load() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([1.0, 0.0]).unwrap(),
        vertex!([0.0, 1.0]).unwrap(),
    ];
    let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel(&RobustKernel::new())
            .unwrap();

    let manifest: DelaunayCheckpointManifest = triangulation.checkpoint_manifest().unwrap();
    assert_eq!(DELAUNAY_CHECKPOINT_SCHEMA_VERSION, 2);
    assert_eq!(
        manifest.manifest_version,
        DELAUNAY_CHECKPOINT_MANIFEST_VERSION
    );
    assert_eq!(manifest.dimension, 2);
    assert_eq!(manifest.f_vector, vec![3, 3, 1]);
    assert_eq!(manifest.euler_characteristic, 1);
    assert_eq!(manifest.digest.version, DELAUNAY_CHECKPOINT_DIGEST_VERSION);
    assert_eq!(
        manifest.digest.algorithm.as_str(),
        DELAUNAY_CHECKPOINT_DIGEST_ALGORITHM
    );
    triangulation.verify_checkpoint_manifest(&manifest).unwrap();
    let mut wrong_dimension = manifest.clone();
    wrong_dimension.dimension = 3;
    assert_eq!(
        triangulation.verify_checkpoint_manifest(&wrong_dimension),
        Err(DelaunayCheckpointError::DimensionMismatch {
            expected: 2,
            actual: 3,
        })
    );

    let checkpoint = to_value(triangulation).unwrap();
    assert_eq!(checkpoint["manifest"], to_value(&manifest).unwrap());
    let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        from_value(checkpoint).unwrap();
    assert_eq!(restored.checkpoint_manifest().unwrap(), manifest);
    restored.validate().unwrap();
}
