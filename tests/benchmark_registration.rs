//! Static contracts for production benchmark registration.

#![forbid(unsafe_code)]

#[test]
fn production_profiling_benchmarks_do_not_substitute_empty_work() {
    let source = include_str!("../benches/profiling_suite.rs");

    assert!(
        !source.contains("b.iter(|| {})"),
        "production benchmark setup failures must abort registration instead of timing empty work"
    );
}
