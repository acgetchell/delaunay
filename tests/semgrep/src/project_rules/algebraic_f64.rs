#![allow(dead_code)]

type BinaryF64Operation = fn(f64, f64) -> f64;

pub fn forbidden_f64_receiver_operations(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        left.algebraic_add(right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        left.algebraic_sub(right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        left.algebraic_mul(right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        left.algebraic_div(right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        left.algebraic_rem(right),
    ]
}

pub fn forbidden_f64_associated_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_add(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_sub(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_mul(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_div(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_rem(left, right),
    ]
}

pub fn forbidden_f64_function_items() -> [BinaryF64Operation; 5] {
    [
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_add,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_sub,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_mul,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_div,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        f64::algebraic_rem,
    ]
}

pub fn forbidden_f64_qualified_calls(left: f64, right: f64) -> [f64; 5] {
    [
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_add(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_sub(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_mul(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_div(left, right),
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_rem(left, right),
    ]
}

pub fn forbidden_f64_qualified_function_items() -> [BinaryF64Operation; 5] {
    [
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_add,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_sub,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_mul,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_div,
        // ruleid: delaunay.rust.no-algebraic-f64-operations
        <f64>::algebraic_rem,
    ]
}

pub fn forbidden_f64_alias() -> BinaryF64Operation {
    // ruleid: delaunay.rust.no-algebraic-f64-operations
    let operation = f64::algebraic_add;
    operation
}

pub fn forbidden_f64_callback(values: &[f64]) -> Option<f64> {
    // ruleid: delaunay.rust.no-algebraic-f64-operations
    values.iter().copied().reduce(f64::algebraic_mul)
}

pub fn permitted_f64_operations(left: f64, right: f64) -> [f64; 6] {
    [
        // ok: delaunay.rust.no-algebraic-f64-operations
        left + right,
        // ok: delaunay.rust.no-algebraic-f64-operations
        left - right,
        // ok: delaunay.rust.no-algebraic-f64-operations
        left * right,
        // ok: delaunay.rust.no-algebraic-f64-operations
        left / right,
        // ok: delaunay.rust.no-algebraic-f64-operations
        left % right,
        // ok: delaunay.rust.no-algebraic-f64-operations
        left.mul_add(right, 1.0),
    ]
}

pub fn permitted_f64_fma_forms(
    left: f64,
    right: f64,
) -> (f64, BinaryF64Operation, fn(f64, f64, f64) -> f64) {
    fn ordinary_add(left: f64, right: f64) -> f64 {
        left + right
    }

    (
        // ok: delaunay.rust.no-algebraic-f64-operations
        <f64>::mul_add(left, right, 1.0),
        // ok: delaunay.rust.no-algebraic-f64-operations
        ordinary_add,
        // ok: delaunay.rust.no-algebraic-f64-operations
        <f64>::mul_add,
    )
}

pub fn permitted_f32_receiver(left: f32, right: f32) -> f32 {
    // ok: delaunay.rust.no-algebraic-f64-operations
    left.algebraic_add(right)
}
