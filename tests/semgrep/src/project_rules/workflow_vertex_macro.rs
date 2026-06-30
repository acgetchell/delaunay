#![allow(dead_code)]

pub struct Vertex<U, const D: usize> {
    data: Option<U>,
}

pub struct VertexValidationError;

pub mod delaunay {
    pub mod prelude {
        pub use crate::Vertex;
    }
}

impl<U, const D: usize> Vertex<U, D> {
    pub fn try_new(_coords: [f64; D]) -> Result<Self, VertexValidationError> {
        Ok(Self { data: None })
    }

    pub fn try_new_with_data(_coords: [f64; D], data: U) -> Result<Self, VertexValidationError> {
        Ok(Self { data: Some(data) })
    }
}

pub fn workflow_vertex_constructor_bad() -> Result<Vertex<(), 3>, VertexValidationError> {
    // ruleid: delaunay.rust.prefer-vertex-macro-for-workflow-fixtures
    Vertex::<(), _>::try_new([0.0, 0.0, 0.0])
}

pub fn workflow_vertex_macro_ok() -> Result<Vertex<(), 3>, VertexValidationError> {
    // ok: delaunay.rust.prefer-vertex-macro-for-workflow-fixtures
    vertex![0.0, 0.0, 0.0]
}

pub fn workflow_vertex_text_ok() -> &'static str {
    // ok: delaunay.rust.prefer-vertex-macro-for-workflow-fixtures
    "Vertex::<(), _>::try_new([0.0, 0.0, 0.0])"
}

pub fn workflow_vertex_constructor_with_data_bad() -> Result<Vertex<u8, 3>, VertexValidationError> {
    // ruleid: delaunay.rust.prefer-vertex-macro-for-workflow-fixtures
    Vertex::<u8, _>::try_new_with_data([0.0, 0.0, 0.0], 1)
}

pub fn workflow_vertex_qualified_constructor_bad() -> Result<Vertex<(), 3>, VertexValidationError> {
    // ruleid: delaunay.rust.prefer-vertex-macro-for-workflow-fixtures
    delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])
}
