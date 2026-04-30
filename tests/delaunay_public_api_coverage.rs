//! Public API integration coverage for `DelaunayTriangulation`.

#![forbid(unsafe_code)]

use delaunay::prelude::geometry::AdaptiveKernel;
use delaunay::prelude::triangulation::{
    ConstructionOptions, DedupPolicy, DelaunayTriangulation,
    DelaunayTriangulationConstructionError, InsertionOrderStrategy, RetryPolicy, TopologyGuarantee,
};
use delaunay::vertex;
#[cfg(feature = "diagnostics")]
use rand::{RngExt, SeedableRng, rngs::StdRng};

type Dt<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

#[test]
fn topology_options_smoke_3d() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let options = ConstructionOptions::default()
        .with_dedup_policy(DedupPolicy::Exact)
        .with_insertion_order(InsertionOrderStrategy::Input)
        .with_retry_policy(RetryPolicy::Disabled);

    let dt = Dt::<3>::with_topology_guarantee_and_options(
        &AdaptiveKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .expect("3D construction with explicit options should succeed");

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 1);
    assert!(dt.validate().is_ok());
}

#[test]
fn statistics_default_on_preprocess_error() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let options = ConstructionOptions::default().with_dedup_policy(DedupPolicy::Epsilon {
        tolerance: f64::NAN,
    });

    let result = Dt::<3>::with_topology_guarantee_and_options_with_construction_statistics(
        &AdaptiveKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    );

    let err = result.expect_err("NaN epsilon should fail during preprocessing");
    assert_eq!(err.statistics.inserted, 0);
    assert_eq!(err.statistics.total_skipped(), 0);
    assert_eq!(err.statistics.total_attempts, 0);
    assert!(err.statistics.skip_samples.is_empty());
    assert!(matches!(
        err.error,
        DelaunayTriangulationConstructionError::Triangulation(_)
    ));
}

#[test]
#[allow(deprecated)]
fn as_triangulation_mut_valid_view() {
    let vertices = vec![
        vertex!([0.0, 0.0]),
        vertex!([1.0, 0.0]),
        vertex!([0.0, 1.0]),
    ];
    let mut dt: DelaunayTriangulation<_, (), (), 2> =
        DelaunayTriangulation::new(&vertices).expect("2D construction should succeed");

    dt.insert(vertex!([0.2, 0.2]))
        .expect("interior insertion should succeed");

    let tri = dt.as_triangulation_mut();
    assert!(tri.is_valid().is_ok());
}

/// Slow search helper to find a natural stale-key repro case.
///
/// This remains ignored by default because it is nondeterministic and expensive.
/// For deterministic coverage, see the forced heuristic rebuild tests in
/// `src/triangulation/delaunay.rs`.
#[cfg(feature = "diagnostics")]
#[test]
#[ignore = "manual search helper; run explicitly to discover natural repro cases"]
fn find_stale_key_after_rebuild() {
    const DIM: usize = 4;
    const INITIAL_COUNT: usize = 12;
    const CASES: usize = 2_000;

    // This probes for a configuration that triggers a heuristic rebuild during automatic
    // flip-repair after insertion, which historically could invalidate the returned VertexKey.
    let mut rng = StdRng::seed_from_u64(0xD3_1A_7A_1C_0A_17_u64);

    for case in 0..CASES {
        let mut vertices = Vec::with_capacity(INITIAL_COUNT);
        for _ in 0..INITIAL_COUNT {
            // Use a coarse lattice + tiny noise to encourage near-degenerate configurations.
            let mut coords = [0.0_f64; DIM];
            for c in &mut coords {
                let base: i32 = rng.random_range(-3..=3);
                let noise: f64 = rng.random_range(-1.0e-6..=1.0e-6);
                *c = f64::from(base) + noise;
            }
            vertices.push(vertex!(coords));
        }

        let Ok(mut dt) = Dt::<4>::new(&vertices) else {
            continue;
        };

        let mut inserted_coords = [0.0_f64; DIM];
        for c in &mut inserted_coords {
            let base: i32 = rng.random_range(-3..=3);
            let noise: f64 = rng.random_range(-1.0e-6..=1.0e-6);
            *c = f64::from(base) + noise;
        }
        let inserted = vertex!(inserted_coords);
        let inserted_uuid = inserted.uuid();

        let Ok(vertex_key) = dt.insert(inserted) else {
            continue;
        };

        let found = dt
            .tds()
            .get_vertex_by_key(vertex_key)
            .is_some_and(|v| v.uuid() == inserted_uuid);

        if found {
            continue;
        }

        #[cfg(feature = "diagnostics")]
        {
            tracing::debug!(case, "FOUND stale key after insertion");
            tracing::debug!(vertex_key = ?vertex_key, inserted_uuid = %inserted_uuid);
            tracing::debug!("initial vertices:");
            for v in &vertices {
                tracing::debug!(coords = ?v.point().coords(), "  vertex");
            }
            tracing::debug!("inserted vertex coords: {inserted_coords:?}");
        }
        panic!("stale VertexKey returned from insert() after heuristic rebuild");
    }

    panic!("no stale-key case found after {CASES} attempts");
}
