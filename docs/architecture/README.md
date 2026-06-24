# Architecture Guidance Index

This directory contains focused architecture references. Keep this file as a
navigation index only; `../code_organization.md` remains the required-reading
architecture hub for agents.

## Documents

| File | Owns |
|-----|-----|
| [`project_structure.md`](project_structure.md) | Repository tree, packaging shape, and top-level directory ownership |
| [`module_map.md`](module_map.md) | `src/` module ownership, layer boundaries, public namespace policy, and architecture principles |
| [`prelude_reference.md`](prelude_reference.md) | Focused prelude taxonomy and import guidance |
| [`module_patterns.md`](module_patterns.md) | In-file Rust organization, section ordering, imports, and test-module layout |

## Maintenance

- Add new architecture guidance here only as a link and one-line ownership note.
- Prefer updating the focused document that owns a surface over growing this
  index.
- Update [`../code_organization.md`](../code_organization.md) when the required
  architecture reading path changes.
- Keep operational workflow and command guidance under [`../dev/`](../dev/).
