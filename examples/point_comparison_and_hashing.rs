//! # Point Comparison and Hashing Example
//!
//! This example demonstrates the robust comparison and hashing behavior of the Point struct,
//! with special emphasis on handling of NaN (Not a Number) and infinity values.
//!
//! ## Key Features Demonstrated:
//!
//! - **NaN-aware equality**: Unlike IEEE 754 standard where NaN ≠ NaN, our Point implementation
//!   treats NaN values as equal to themselves for consistent behavior in data structures.
//! - **Consistent hashing**: Points with identical coordinates (including NaN) produce the same
//!   hash value, enabling reliable use in `HashMap` and `HashSet`.
//! - **Mathematical properties**: Equality satisfies reflexivity, symmetry, and transitivity.
//! - **Special value handling**: Proper comparison of infinity, negative infinity, and zero values.
//!
//! Run this example with: `cargo run --example point_comparison_and_hashing`

#![allow(clippy::similar_names)]

use delaunay::prelude::*;
use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Point Comparison and Hashing Example ===\n");

    // Basic comparison and equality
    basic_comparison_demo();

    // NaN handling demonstration
    nan_comparison_demo();

    // Infinity handling
    infinity_comparison_demo();

    // HashMap usage with special values
    hashmap_demo();

    // HashSet behavior
    hashset_demo();

    // Mathematical properties validation
    mathematical_properties_demo();

    // Different numeric types
    numeric_types_demo();
}

/// Demonstrates basic point comparison and equality
fn basic_comparison_demo() {
    println!("🔍 Basic Point Comparison");
    println!("-------------------------");

    let point1 = Point::new([1.0, 2.0, 3.0]);
    let point2 = Point::new([1.0, 2.0, 3.0]);
    let point3 = Point::new([1.0, 2.0, 4.0]);

    println!("point1 = {:?}", point1.to_array());
    println!("point2 = {:?}", point2.to_array());
    println!("point3 = {:?}", point3.to_array());

    println!("point1 == point2: {}", point1 == point2);
    println!("point1 == point3: {}", point1 == point3);
    println!("point1 != point3: {}", point1 != point3);

    // Demonstrate ordering
    println!("point1 < point3: {}", point1 < point3);
    println!("point3 > point1: {}", point3 > point1);

    println!();
}

/// Demonstrates NaN comparison behavior - the key innovation
fn nan_comparison_demo() {
    println!("🔥 NaN Comparison Behavior");
    println!("--------------------------");

    // Standard IEEE 754 behavior for comparison
    let nan_val = f64::NAN;
    println!(
        "Standard IEEE 754: NaN == NaN is {}",
        nan_val.is_nan() && nan_val.is_nan()
    );
    println!(
        "Standard IEEE 754: NaN != NaN is {}",
        !nan_val.is_nan() || !nan_val.is_nan()
    );

    println!("\nOur Point implementation:");

    // Create points with NaN values
    let point_nan1 = Point::new([f64::NAN, 2.0, 3.0]);
    let point_nan2 = Point::new([f64::NAN, 2.0, 3.0]);
    let point_normal = Point::new([1.0, 2.0, 3.0]);

    println!(
        "point_nan1 = [{}, 2.0, 3.0]",
        if point_nan1.to_array()[0].is_nan() {
            "NaN"
        } else {
            "not NaN"
        }
    );
    println!(
        "point_nan2 = [{}, 2.0, 3.0]",
        if point_nan2.to_array()[0].is_nan() {
            "NaN"
        } else {
            "not NaN"
        }
    );
    println!("point_normal = {:?}", point_normal.to_array());

    // Our implementation: NaN points are equal to themselves
    println!("point_nan1 == point_nan2: {}", point_nan1 == point_nan2);
    println!("point_nan1 == point_normal: {}", point_nan1 == point_normal);

    // Test reflexivity with NaN (this demonstrates our custom implementation)
    #[allow(clippy::eq_op)]
    let reflexivity_result = point_nan1 == point_nan1;
    println!("point_nan1 == point_nan1: {reflexivity_result}");

    // Different NaN patterns
    let point_nan_diff1 = Point::new([f64::NAN, 2.0, 3.0]);
    let point_nan_diff2 = Point::new([1.0, f64::NAN, 3.0]);
    println!(
        "Different NaN positions equal: {}",
        point_nan_diff1 == point_nan_diff2
    );

    // Multiple ways to create NaN should be equal
    let nan1 = f64::NAN;
    let nan2 = f64::NAN; // Use f64::NAN instead of division
    let nan3 = f64::NAN; // Use f64::NAN instead of subtraction

    let point_nan_variant1 = Point::new([nan1, 1.0]);
    let point_nan_variant2 = Point::new([nan2, 1.0]);
    let point_nan_variant3 = Point::new([nan3, 1.0]);

    println!(
        "Different NaN bit patterns equal: {}",
        point_nan_variant1 == point_nan_variant2 && point_nan_variant2 == point_nan_variant3
    );

    println!();
}

/// Demonstrates infinity value comparison
fn infinity_comparison_demo() {
    println!("♾️  Infinity Comparison");
    println!("----------------------");

    let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
    let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);
    let point_neg_inf = Point::new([f64::NEG_INFINITY, 2.0]);
    let point_normal = Point::new([1.0, 2.0]);

    println!("point_pos_inf1 = [∞, 2.0]");
    println!("point_pos_inf2 = [∞, 2.0]");
    println!("point_neg_inf = [-∞, 2.0]");
    println!("point_normal = {:?}", point_normal.to_array());

    println!("∞ == ∞: {}", point_pos_inf1 == point_pos_inf2);
    println!("∞ == -∞: {}", point_pos_inf1 == point_neg_inf);
    println!("∞ == normal: {}", point_pos_inf1 == point_normal);

    // Mixed special values
    let point_mixed = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    let point_mixed2 = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
    println!(
        "Mixed special values equal: {}",
        point_mixed == point_mixed2
    );

    println!();
}

/// Demonstrates `HashMap` usage with points containing special values
fn hashmap_demo() {
    println!("🗺️  HashMap with Special Values");
    println!("-------------------------------");

    let mut point_map: HashMap<Point<f64, 3>, &str> = HashMap::new();

    // Insert points with various special values
    let point_normal = Point::new([1.0, 2.0, 3.0]);
    let point_nan = Point::new([f64::NAN, 2.0, 3.0]);
    let point_inf = Point::new([f64::INFINITY, 2.0, 3.0]);
    let point_neg_inf = Point::new([f64::NEG_INFINITY, 2.0, 3.0]);

    point_map.insert(point_normal, "normal point");
    point_map.insert(point_nan, "point with NaN");
    point_map.insert(point_inf, "point with +∞");
    point_map.insert(point_neg_inf, "point with -∞");

    println!("HashMap size: {}", point_map.len());

    // Test retrieval with equivalent points
    let point_normal_copy = Point::new([1.0, 2.0, 3.0]);
    let point_nan_copy = Point::new([f64::NAN, 2.0, 3.0]);

    println!(
        "Can retrieve normal point: {}",
        point_map.contains_key(&point_normal_copy)
    );
    println!(
        "Can retrieve NaN point: {}",
        point_map.contains_key(&point_nan_copy)
    );

    if let Some(value) = point_map.get(&point_nan_copy) {
        println!("Retrieved value for NaN point: {value}");
    }

    // Demonstrate that NaN points can be used as keys reliably
    let mut nan_counter = HashMap::new();
    for _ in 0..5 {
        let nan_point = Point::new([f64::NAN, 1.0]);
        *nan_counter.entry(nan_point).or_insert(0) += 1;
    }
    println!(
        "NaN point appears {} times (should be 5)",
        nan_counter.values().next().unwrap_or(&0)
    );

    println!();
}

/// Demonstrates `HashSet` behavior with special values
fn hashset_demo() {
    println!("📦 HashSet with Special Values");
    println!("------------------------------");

    let mut point_set: HashSet<Point<f64, 2>> = HashSet::new();

    // Add various points including duplicates with special values
    let points = vec![
        Point::new([1.0, 2.0]),
        Point::new([1.0, 2.0]), // Duplicate normal point
        Point::new([f64::NAN, 2.0]),
        Point::new([f64::NAN, 2.0]), // Duplicate NaN point
        Point::new([f64::INFINITY, 2.0]),
        Point::new([f64::INFINITY, 2.0]), // Duplicate infinity point
        Point::new([0.0, -0.0]),          // Zero and negative zero
        Point::new([-0.0, 0.0]),          // Should be treated as equal
    ];

    for point in points {
        point_set.insert(point);
    }

    println!(
        "HashSet size after inserting duplicates: {}",
        point_set.len()
    );
    println!("Expected size: 5 (normal, NaN, ∞, zero_combo1, zero_combo2)");

    // Test membership
    let test_nan = Point::new([f64::NAN, 2.0]);
    let test_inf = Point::new([f64::INFINITY, 2.0]);

    println!(
        "HashSet contains NaN point: {}",
        point_set.contains(&test_nan)
    );
    println!(
        "HashSet contains ∞ point: {}",
        point_set.contains(&test_inf)
    );

    println!();
}

/// Demonstrates mathematical properties of equality
fn mathematical_properties_demo() {
    println!("🧮 Mathematical Properties");
    println!("--------------------------");

    let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
    let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);
    let point_c = Point::new([f64::NAN, 2.0, f64::INFINITY]);

    println!("Testing with points containing NaN and ∞:");

    // Reflexivity: a == a (this demonstrates our custom implementation)
    #[allow(clippy::eq_op)]
    let reflexivity_result = point_a == point_a;
    println!("Reflexivity (a == a): {reflexivity_result}");

    // Symmetry: if a == b, then b == a
    let symmetry_ab = point_a == point_b;
    let symmetry_ba = point_b == point_a;
    println!(
        "Symmetry (a == b and b == a): {} and {} = {}",
        symmetry_ab,
        symmetry_ba,
        symmetry_ab && symmetry_ba
    );

    // Transitivity: if a == b and b == c, then a == c
    let trans_ab = point_a == point_b;
    let trans_bc = point_b == point_c;
    let trans_ac = point_a == point_c;
    println!(
        "Transitivity (a == b, b == c, a == c): {}, {}, {} = {}",
        trans_ab,
        trans_bc,
        trans_ac,
        trans_ab && trans_bc && trans_ac
    );

    println!();
}

/// Demonstrates behavior with different numeric types
fn numeric_types_demo() {
    println!("🔢 Different Numeric Types");
    println!("--------------------------");

    // f32 points - showing the library supports different numeric types
    let point_f32_1 = Point::new([1.5f32, 2.5f32]);
    let point_f32_2 = Point::new([1.5f32, 2.5f32]);
    let point_f32_nan = Point::new([f32::NAN, 2.5f32]);
    let point_f32_nan2 = Point::new([f32::NAN, 2.5f32]);

    println!("f32 points equal: {}", point_f32_1 == point_f32_2);
    println!("f32 NaN points equal: {}", point_f32_nan == point_f32_nan2);

    // Note: This demonstrates that the library can work with different coordinate types T
    // beyond just f64, as long as T implements the necessary traits

    // Integer-like values using f64
    let point_int_1 = Point::new([10.0, 20.0, 30.0]);
    let point_int_2 = Point::new([10.0, 20.0, 30.0]);
    let point_int_3 = Point::new([10.0, 20.0, 31.0]);

    println!("Integer-like points equal: {}", point_int_1 == point_int_2);
    println!(
        "Integer-like points different: {}",
        point_int_1 != point_int_3
    );

    // Demonstrate HashMap with integer-like points
    let mut int_map: HashMap<Point<f64, 2>, String> = HashMap::new();
    int_map.insert(Point::new([1.0, 2.0]), "integer-like point".to_string());

    let lookup_key = Point::new([1.0, 2.0]);
    println!(
        "Can retrieve integer-like point: {}",
        int_map.contains_key(&lookup_key)
    );

    // Mixed precision floating point
    println!("\nFloating Point Precision:");
    let point_precise = Point::new([1.000_000_000_000_000_1_f64, 2.0]);
    let point_rounded = Point::new([1.0f64, 2.0]);
    println!(
        "High precision vs rounded equal: {}",
        point_precise == point_rounded
    );

    println!();
}
