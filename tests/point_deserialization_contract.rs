//! Format-independent proof that deserialization preserves the Point boundary.

#![forbid(unsafe_code)]

use delaunay::prelude::geometry::{
    Coordinate, CoordinateValidationError, InvalidCoordinateValue, Point,
};
use serde::de::{self, DeserializeSeed, SeqAccess, Visitor, value::SeqAccessDeserializer};
use serde::{Deserialize, Deserializer};

/// A numeric decoder that deliberately cannot infer a value's type.
struct TypedCoordinate(f64);

impl<'de> Deserializer<'de> for TypedCoordinate {
    type Error = de::value::Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        Err(de::Error::custom(
            "coordinate decoding requires an explicit type",
        ))
    }

    fn deserialize_f64<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_f64(self.0)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_unit()
    }

    fn is_human_readable(&self) -> bool {
        false
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier
    }
}

/// Unlike a codec's tuple decoder, this sequence does not enforce tuple length.
struct TypedCoordinates<'a>(std::slice::Iter<'a, f64>);

impl<'de> SeqAccess<'de> for TypedCoordinates<'_> {
    type Error = de::value::Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Self::Error>
    where
        T: DeserializeSeed<'de>,
    {
        self.0
            .next()
            .map(|&value| seed.deserialize(TypedCoordinate(value)))
            .transpose()
    }
}

fn decode_typed<const D: usize>(coordinates: &[f64]) -> Result<Point<D>, de::value::Error> {
    Point::deserialize(SeqAccessDeserializer::new(TypedCoordinates(
        coordinates.iter(),
    )))
}

fn assert_numeric_boundary<const D: usize>() {
    let values = [
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        1.25,
        -2.5,
    ];
    assert_eq!(Point::<D>::default().to_array().map(f64::to_bits), [0; D]);
    for offset in 0..values.len() {
        let coordinates: [f64; D] =
            std::array::from_fn(|index| values[(index + offset) % values.len()]);
        let expected = Point::try_new(coordinates).expect("all fixture values are finite");
        let converted = Point::try_from(coordinates).expect("finite f64 conversion must succeed");
        let decoded = decode_typed::<D>(&coordinates).expect("typed coordinates must deserialize");
        let expected_bits = expected.to_array().map(f64::to_bits);
        assert_eq!(converted.to_array().map(f64::to_bits), expected_bits);
        assert_eq!(decoded.to_array().map(f64::to_bits), expected_bits);

        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&expected, &mut bytes).expect("point must serialize as CBOR");
        let round_trip: Point<D> =
            ciborium::de::from_reader(bytes.as_slice()).expect("CBOR point must round-trip");
        assert_eq!(round_trip.to_array().map(f64::to_bits), expected_bits);
    }

    for coordinate_index in 0..D {
        for (raw, coordinate_value) in [
            (f64::NAN, InvalidCoordinateValue::Nan),
            (f64::INFINITY, InvalidCoordinateValue::PositiveInfinity),
            (f64::NEG_INFINITY, InvalidCoordinateValue::NegativeInfinity),
        ] {
            let mut coordinates = [0.0; D];
            coordinates[coordinate_index] = raw;
            let expected = CoordinateValidationError::InvalidCoordinate {
                coordinate_index,
                coordinate_value,
                dimension: D,
            };
            assert_eq!(Point::try_new(coordinates), Err(expected.clone()));
            let error = decode_typed::<D>(&coordinates)
                .expect_err("non-finite coordinates must not bypass the constructor");
            assert_eq!(error.to_string(), expected.to_string());
        }
    }

    for length in [D.saturating_sub(1), D + 1] {
        if length == D {
            continue;
        }
        let error = decode_typed::<D>(&vec![0.0; length])
            .expect_err("the point parser must enforce its own coordinate count");
        assert_eq!(
            error.to_string(),
            format!("invalid length {length}, expected an array of {D} finite numeric coordinates")
        );
    }
}

#[test]
fn point_numeric_deserialization_contract_2d() {
    assert_numeric_boundary::<2>();
}

#[test]
fn point_numeric_deserialization_contract_3d() {
    assert_numeric_boundary::<3>();
}

#[test]
fn point_numeric_deserialization_contract_4d() {
    assert_numeric_boundary::<4>();
}

#[test]
fn point_numeric_deserialization_contract_5d() {
    assert_numeric_boundary::<5>();
}

#[test]
fn point_numeric_deserialization_contract_low_dimensions() {
    assert_numeric_boundary::<0>();
    assert_numeric_boundary::<1>();
}

#[test]
fn point_json_rejects_non_numeric_and_wrong_arity_values() {
    for json in [
        "[]",
        "[1]",
        "[1,2,3]",
        "[null,2]",
        "[true,2]",
        r#"["1",2]"#,
        r#"["NaN",2]"#,
        r#"["Infinity",2]"#,
        "[{},2]",
        "[[],2]",
    ] {
        let error = serde_json::from_str::<Point<2>>(json)
            .expect_err("only exactly two numeric coordinates are accepted");
        assert_eq!(
            error.classify(),
            serde_json::error::Category::Data,
            "{json}"
        );
    }

    let point: Point<2> = serde_json::from_str("[1,-2.5]").expect("numeric JSON coordinates");
    assert_eq!(point, Point::try_new([1.0, -2.5]).unwrap());
    assert_eq!(serde_json::to_string(&point).unwrap(), "[1.0,-2.5]");
}
