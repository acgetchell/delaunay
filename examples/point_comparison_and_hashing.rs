//! # Point Comparison and Hashing Example
//!
//! Demonstrates finite [`Point`] comparison, hashing, and validation failures for
//! non-finite input coordinates.
//!
//! Run with: `cargo run --example point_comparison_and_hashing`

use delaunay::prelude::geometry::{CoordinateValidationError, InvalidCoordinateValue, Point};
use std::collections::{HashMap, HashSet};

fn main() -> Result<(), CoordinateValidationError> {
    println!("=== Point Comparison and Hashing Example ===\n");

    basic_comparison_demo()?;
    hashmap_demo()?;
    hashset_demo()?;
    validation_demo();

    Ok(())
}

fn basic_comparison_demo() -> Result<(), CoordinateValidationError> {
    println!("Basic point comparison");
    println!("----------------------");

    let point1 = Point::try_new([1.0, 2.0, 3.0])?;
    let point2 = Point::try_new([1.0, 2.0, 3.0])?;
    let point3 = Point::try_new([1.0, 2.0, 4.0])?;

    println!("point1 = {:?}", point1.coords());
    println!("point2 = {:?}", point2.coords());
    println!("point3 = {:?}", point3.coords());
    println!("point1 == point2: {}", point1 == point2);
    println!("point1 == point3: {}", point1 == point3);
    println!("point1 < point3: {}", point1 < point3);
    println!();

    Ok(())
}

fn hashmap_demo() -> Result<(), CoordinateValidationError> {
    println!("HashMap with finite points");
    println!("--------------------------");

    let mut point_map: HashMap<Point<3>, &str> = HashMap::new();
    let point = Point::try_new([1.0, 2.0, 3.0])?;
    let same_point = Point::try_new([1.0, 2.0, 3.0])?;
    let other_point = Point::try_new([2.0, 3.0, 4.0])?;

    point_map.insert(point, "first point");
    point_map.insert(other_point, "second point");

    println!("HashMap size: {}", point_map.len());
    println!(
        "Can retrieve equivalent key: {}",
        point_map.contains_key(&same_point)
    );
    println!();

    Ok(())
}

fn hashset_demo() -> Result<(), CoordinateValidationError> {
    println!("HashSet duplicate handling");
    println!("--------------------------");

    let mut point_set: HashSet<Point<2>> = HashSet::new();
    for point in [
        Point::try_new([1.0, 2.0])?,
        Point::try_new([1.0, 2.0])?,
        Point::try_new([0.0, -0.0])?,
        Point::try_new([-0.0, 0.0])?,
        Point::try_new([3.0, 4.0])?,
    ] {
        point_set.insert(point);
    }

    println!("HashSet size after duplicates: {}", point_set.len());
    println!("Expected size: 3");
    println!();

    Ok(())
}

fn validation_demo() {
    println!("Non-finite input validation");
    println!("---------------------------");

    for (label, result) in [
        ("NaN", Point::<2>::try_new([f64::NAN, 1.0])),
        ("+infinity", Point::<2>::try_new([f64::INFINITY, 1.0])),
        ("-infinity", Point::<2>::try_new([f64::NEG_INFINITY, 1.0])),
    ] {
        match result {
            Ok(point) => println!("{label}: unexpectedly accepted {:?}", point.coords()),
            Err(CoordinateValidationError::InvalidCoordinate {
                coordinate_value, ..
            }) => {
                let kind = match coordinate_value {
                    InvalidCoordinateValue::Nan => "NaN",
                    InvalidCoordinateValue::PositiveInfinity => "+infinity",
                    InvalidCoordinateValue::NegativeInfinity => "-infinity",
                    _ => "non-finite value",
                };
                println!("{label}: rejected {kind}");
            }
            Err(_) => println!("{label}: rejected invalid coordinate"),
        }
    }
}
