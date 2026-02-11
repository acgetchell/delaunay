//! Regression test for a deterministic 3D point set that previously triggered
//! non-convergent global flip repair during batch construction.

use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::{ConstructionOptions, DelaunayTriangulation, RetryPolicy};
use delaunay::vertex;
use std::num::NonZeroUsize;

#[test]
fn batch_construction_succeeds_for_seed_173_count_50_3d() {
    let bounds = (-100.0, 100.0);
    let seed = 173_u64;
    let count = 50_usize;

    let points = generate_random_points_seeded::<f64, 3>(count, bounds, seed)
        .expect("generate_random_points_seeded failed");
    let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();

    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: NonZeroUsize::new(6).expect("retry attempts must be non-zero"),
        base_seed: Some(seed),
    });

    let _dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_options(&vertices, options)
            .expect("batch construction failed");
}
