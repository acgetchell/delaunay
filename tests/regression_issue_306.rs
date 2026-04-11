//! Regression test for issue #306: 3D flip-repair cycling in release builds.
//!
//! This test lives in an integration test crate (not `#[cfg(test)]` unit tests)
//! so the library is compiled *without* `cfg(test)`.  This is essential because
//! the original bug used `cfg(any(test, debug_assertions))` to gate repair
//! constants — unit tests would always see the permissive values even in
//! `--release` mode.

use delaunay::geometry::util::generate_random_points_in_ball_seeded;
use delaunay::prelude::triangulation::*;

/// The 35-vertex 3D seed `0xE30C78582376677C` produces a Hilbert-ordered
/// insertion sequence where vertex 23 triggers flip-repair cycling on
/// co-spherical configurations.
///
/// With the former release-mode `MAX_REPEAT_SIGNATURE = 32` and
/// `RetryPolicy::Disabled`, construction failed deterministically.
/// The fix (#306) unified these constants so the repair has sufficient
/// patience and shuffled retries are always available.
///
/// Run with `cargo test --release` to exercise the release profile.
#[test]
fn regression_issue_306_3d_construction_succeeds() {
    let seed: u64 = 0xE30C_7858_2376_677C;
    let points = generate_random_points_in_ball_seeded::<f64, 3>(35, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 3>> = points.into_iter().map(|p| vertex!(p)).collect();

    let dt: Result<DelaunayTriangulation<_, (), (), 3>, _> = DelaunayTriangulation::new(&vertices);
    assert!(
        dt.is_ok(),
        "35-vertex 3D construction with seed 0x{seed:X} should succeed \
         (requires unified repair constants); got: {}",
        dt.unwrap_err()
    );
}
