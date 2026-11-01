#![forbid(unsafe_code)]

use delaunay::prelude::checkpoint::{
    DelaunayCheckpoint, DelaunayCheckpointCodecError, DelaunayCheckpointError,
    DelaunayCheckpointLoadError,
};
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, vertex,
};
use delaunay::prelude::geometry::RobustKernel;
use serde_json::{Value, from_slice, from_value, to_vec};
use std::{error::Error, io::ErrorKind};

/// Exercise the public load boundary without the parent crate's dev dependencies.
fn codec_error(checkpoint: &Value, bytes: &[u8]) -> DelaunayCheckpointCodecError {
    let mut checkpoint = checkpoint.clone();
    checkpoint["tds"] = serde_json::json!(bytes);
    let error = from_value::<DelaunayCheckpoint<(), (), 2>>(checkpoint)
        .unwrap()
        .try_into_delaunay()
        .unwrap_err();
    let DelaunayCheckpointLoadError::Checkpoint {
        source: DelaunayCheckpointError::TdsCodec { source },
    } = &error
    else {
        panic!("expected a typed embedded-CBOR error, got {error:?}");
    };
    assert_eq!(
        error
            .source()
            .and_then(|source| source.downcast_ref::<DelaunayCheckpointCodecError>()),
        Some(source),
    );
    source.clone()
}

/// Pin malformed-image categories when Ciborium's std feature is not unified in.
fn check_codec_errors(json: &[u8]) {
    let checkpoint = from_slice(json).unwrap();
    for (bytes, expected) in [
        (
            b"\xbf".as_slice(),
            DelaunayCheckpointCodecError::Io {
                kind: ErrorKind::UnexpectedEof,
            },
        ),
        (
            b"\x1c".as_slice(),
            DelaunayCheckpointCodecError::Syntax { offset: 0 },
        ),
        (
            b"\xbf\x68vertices\x1c".as_slice(),
            DelaunayCheckpointCodecError::Syntax { offset: 10 },
        ),
    ] {
        assert_eq!(codec_error(&checkpoint, bytes), expected);
    }
    assert!(matches!(
        codec_error(&checkpoint, b"\x01"),
        DelaunayCheckpointCodecError::Semantic { offset: None, detail } if !detail.is_empty()
    ));

    // {"vertices": [{"data": [[[...null...]]]}]} exceeds the decoder's
    // 256-level recursion limit before snapshot validation or payload hydration.
    let mut nested = b"\xa1\x68vertices\x81\xa1\x64data".to_vec();
    nested.extend([0x81; 300]);
    nested.push(0xf6);
    assert_eq!(
        codec_error(&checkpoint, &nested),
        DelaunayCheckpointCodecError::RecursionLimitExceeded,
    );
}

fn main() {
    let vertices = [
        vertex!([0.0, 0.0]).unwrap(),
        vertex!([0.0, -22.546_422_723_221_383]).unwrap(),
        vertex!([12.252_739_760_228_783, 0.0]).unwrap(),
        vertex!([1.572_332_120_964_092_1, 2.883_274_933_268_964]).unwrap(),
    ];
    let triangulation: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> =
        DelaunayTriangulationBuilder::new(&vertices)
            .build_with_kernel(&RobustKernel::new())
            .unwrap();
    let mut expected = triangulation
        .vertices()
        .map(|(_, vertex)| vertex.point().coords().map(f64::to_bits))
        .collect::<Vec<_>>();
    expected.sort_unstable();

    let json = to_vec(&triangulation).unwrap();
    let restored: DelaunayTriangulation<RobustKernel<f64>, (), (), 2> = from_slice(&json).unwrap();
    let mut actual = restored
        .vertices()
        .map(|(_, vertex)| vertex.point().coords().map(f64::to_bits))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    restored.validate().unwrap();
    check_codec_errors(&json);
}
