# Rust Development Guidelines

Rust guidance for this repository is split so agents can load the short routing
page first and then jump to the detailed rules that match the work they are
touching.

Agents modifying Rust code must read the focused reference:

- [`rust/reference.md`](rust/reference.md) - API, invariant, naming, error,
  documentation, implementation, testing, performance, dependency, and lint
  conventions

Keep this file as an index. Move detailed Rust policy into focused files under
`docs/dev/rust/` so `AGENTS.md` and the top-level development index remain
quick to parse.
