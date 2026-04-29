#![allow(dead_code, unused_imports)]

use rustc_hash::{FxHashMap as FastHashMap, FxHashSet as FastHashSet};

// ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
use std::collections::{BTreeMap, HashMap};
// ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
use std::collections::{BTreeSet, HashSet};

// ok: delaunay.rust.no-std-hash-collections-in-hot-src
type GoodMap = FastHashMap<u8, u8>;
// ok: delaunay.rust.no-std-hash-collections-in-hot-src
type GoodSet = FastHashSet<u8>;

// ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
type SlowMap = std::collections::HashMap<u8, u8>;
// ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
type SlowSet = std::collections::HashSet<u8>;

trait SlowSignature {
    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    fn map(&self) -> std::collections::HashMap<u8, u8>;

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    fn set(&self) -> std::collections::HashSet<u8>;
}

fn constructors() {
    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _map = std::collections::HashMap::<u8, u8>::with_capacity(4);

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _set = std::collections::HashSet::<u8>::with_capacity(4);

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _map_default = std::collections::HashMap::<u8, u8>::default();

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _set_default = std::collections::HashSet::<u8>::default();
}

fn collects() {
    let pairs = [(1_u8, 2_u8)];
    let values = [1_u8, 2_u8];

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _map = pairs
        .into_iter()
        .collect::<std::collections::HashMap<u8, u8>>();

    // ruleid: delaunay.rust.no-std-hash-collections-in-hot-src
    let _set = values
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
}
