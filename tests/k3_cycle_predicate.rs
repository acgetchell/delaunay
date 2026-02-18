//! Debug-only predicate inspection for k=3 (3↔3) flips in 4D.
//!
//! This test is ignored by default; run it manually with:
//! `cargo test --test k3_cycle_predicate -- --ignored --nocapture`
//! The sample diagnostic event is emitted at warn level by default.

#![forbid(unsafe_code)]

use delaunay::geometry::kernel::{FastKernel, Kernel, RobustKernel};
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

fn eval_k3_violation_4d<K: Kernel<4>>(
    kernel: &K,
    ridge: &[Point<K::Scalar, 4>; 3],
    tri: &[Point<K::Scalar, 4>; 3],
) -> (bool, [i32; 3]) {
    let mut signs = [0_i32; 3];
    let mut violates = false;

    for (i, missing) in tri.iter().copied().enumerate() {
        let mut simplex: Vec<Point<K::Scalar, 4>> = Vec::with_capacity(5);
        simplex.extend_from_slice(ridge);
        for (j, v) in tri.iter().copied().enumerate() {
            if i != j {
                simplex.push(v);
            }
        }

        let sign = kernel.in_sphere(&simplex, &missing).expect("insphere");
        signs[i] = sign;
        if sign > 0 {
            violates = true;
        }
    }

    (violates, signs)
}

#[test]
#[ignore = "debug-only predicate inspection"]
fn debug_k3_cycle_signature_259165653798390695() {
    init_tracing();

    // From `target/large_scale_debug_4d_box_n25_seed481_cycle_ctx.log`:
    // removed_face=[2,4,14], inserted_face=[3,6,9]
    let p2 = Point::new([
        -61.608_777_833_438_296,
        -66.120_761_244_767_1,
        -70.084_028_705_550_51,
        51.627_062_071_549_4,
    ]);
    let p4 = Point::new([
        -74.525_112_841_041_79,
        -71.179_612_578_192_67,
        -7.779_137_511_431_33,
        98.078_720_495_651_6,
    ]);
    let p14 = Point::new([
        68.478_662_333_984_86,
        61.150_723_102_058_75,
        -79.146_708_282_416_92,
        22.565_165_028_265_085,
    ]);
    let p3 = Point::new([
        -98.436_407_264_857_85,
        -43.518_864_817_972_87,
        -89.175_617_527_619_58,
        97.720_089_025_220_69,
    ]);
    let p6 = Point::new([
        -43.867_795_141_236_87,
        -91.336_198_189_478_38,
        47.778_961_799_523_046,
        -30.440_710_341_391_38,
    ]);
    let p9 = Point::new([
        -25.116_117_520_722_582,
        12.988_805_299_508_769,
        46.522_753_060_923_66,
        56.881_024_670_699_53,
    ]);

    let ridge_a = [p2, p4, p14];
    let tri_a = [p3, p6, p9];

    let ridge_b = tri_a;
    let tri_b = ridge_a;

    let fast = FastKernel::<f64>::new();
    let robust = RobustKernel::<f64>::new();

    let (a_fast, a_fast_signs) = eval_k3_violation_4d(&fast, &ridge_a, &tri_a);
    let (b_fast, b_fast_signs) = eval_k3_violation_4d(&fast, &ridge_b, &tri_b);

    let (a_robust, a_robust_signs) = eval_k3_violation_4d(&robust, &ridge_a, &tri_a);
    let (b_robust, b_robust_signs) = eval_k3_violation_4d(&robust, &ridge_b, &tri_b);

    tracing::warn!(
        signature = 259_165_653_798_390_695_u64,
        a_ridge = ?[2_u8, 4_u8, 14_u8],
        a_tri = ?[3_u8, 6_u8, 9_u8],
        a_fast_violates = a_fast,
        a_fast_signs = ?a_fast_signs,
        a_robust_violates = a_robust,
        a_robust_signs = ?a_robust_signs,
        b_ridge = ?[3_u8, 6_u8, 9_u8],
        b_tri = ?[2_u8, 4_u8, 14_u8],
        b_fast_violates = b_fast,
        b_fast_signs = ?b_fast_signs,
        b_robust_violates = b_robust,
        b_robust_signs = ?b_robust_signs,
        "k=3 cycle predicate sample"
    );
}
