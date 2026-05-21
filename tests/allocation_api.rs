//! Smoke tests for allocation measurement support.
//!
//! These tests run only with `--features count-allocations` because the feature
//! installs the allocation-counting global allocator. Hot-path allocation
//! budgets live in `benches/allocation_hot_paths.rs`.

#![cfg(feature = "count-allocations")]

use delaunay::prelude::query::measure_with_result;

#[test]
fn measure_with_result_reports_zero_for_stack_only_work() {
    let (result, alloc_info) = measure_with_result(|| {
        let values = [1_u8, 2, 3, 4];
        values.iter().copied().sum::<u8>()
    });

    assert_eq!(result, 10);
    assert_eq!(
        alloc_info.count_total, 0,
        "stack-only work should not allocate: {alloc_info:?}"
    );
    assert_eq!(
        alloc_info.bytes_total, 0,
        "stack-only work should allocate zero bytes: {alloc_info:?}"
    );
    assert_eq!(
        alloc_info.count_current, 0,
        "stack-only work should not retain allocations: {alloc_info:?}"
    );
    assert_eq!(
        alloc_info.bytes_current, 0,
        "stack-only work should retain zero bytes: {alloc_info:?}"
    );
}

#[test]
fn measure_with_result_returns_value_and_allocation_info() {
    let (result, alloc_info) = measure_with_result(|| vec![0_u8; 1024]);

    assert_eq!(result.len(), 1024);
    assert!(
        alloc_info.count_total > 0,
        "allocation measurement should record the vector allocation: {alloc_info:?}"
    );
    assert!(
        alloc_info.bytes_total >= 1024,
        "allocation measurement should record at least the vector payload bytes: {alloc_info:?}"
    );
}
