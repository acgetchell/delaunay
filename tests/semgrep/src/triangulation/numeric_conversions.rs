use crate::geometry::util::safe_usize_to_scalar;

fn silent_default(value: usize) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks
    safe_usize_to_scalar::<f64>(value).unwrap_or(0.0)
}

fn silent_lazy_default(value: usize) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks
    safe_usize_to_scalar::<f64>(value).unwrap_or_else(|_| 0.0)
}

fn propagated_conversion(value: usize) -> Result<f64, CoordinateConversionError> {
    // ok: delaunay.rust.no-silent-conversion-fallbacks
    safe_usize_to_scalar::<f64>(value)
}

fn unrelated_default(value: Option<usize>) -> usize {
    // ok: delaunay.rust.no-silent-conversion-fallbacks
    value.unwrap_or(0)
}
