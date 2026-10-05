// ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
use std::collections::HashMap;

// ok: delaunay.rust.no-std-hash-collections-in-hot-src
use crate::core::collections::FastHashMap;

fn unapproved_map() {
    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _values = std::collections::HashMap::<u64, u64>::new();
}

fn approved_map() {
    // ok: delaunay.rust.no-std-hash-collections-in-hot-src
    let _values = FastHashMap::<u64, u64>::default();
}
