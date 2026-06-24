#![allow(dead_code)]

pub struct Vertex<U, const D: usize> {
    data: Option<U>,
}

pub struct VertexValidationError;

impl<U, const D: usize> Vertex<U, D> {
    pub fn try_new(_coords: [f64; D]) -> Result<Self, VertexValidationError> {
        Ok(Self { data: None })
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
