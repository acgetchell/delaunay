# AGENTS.md

Essential guidance for AI assistants working in this repository.

This file is the **entry point for all coding agents**. Detailed rules are
split into additional documents under `docs/dev/`. Agents MUST read the
referenced files before making changes.

---

## Required Reading

Before modifying code, agents MUST read:

- `AGENTS.md` (this file)
- **All files in `docs/dev/*.md`** – repository development rules
- `docs/code_organization.md` – module layout and architecture

The `docs/dev/` directory contains the authoritative development guidance
for this repository. Agents must load every file in that directory before
making changes.

---

## Core Rules

### Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git commands that modify version control state
- **ALLOWED**: read‑only git commands (`git --no-pager status`, `git --no-pager diff`, `git --no-pager log`, `git --no-pager show`, `git --no-pager blame`)
- **ALWAYS** use `git --no-pager` when reading git output
- Suggest git commands that modify version control state for the user to run manually

### Code Editing

- **NEVER** use `sed`, `awk`, `perl`, or `python` to modify code
- **ALWAYS** use the patch editing mechanism provided by the agent
- Shell text tools may be used for **read‑only analysis only**

### Commit Message Generation

When generating commit messages:

1. Run `git --no-pager diff --cached --stat`
2. Use conventional commits: `<type>: <summary>`
3. Valid types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include bullet‑point body describing key changes
5. Present inside a code block so the user can commit manually

#### Changelog‑Aware Body Text

Commit bodies appear **verbatim** in `CHANGELOG.md` (indented by
git‑cliff's template). Write them as clean, readable prose:

- Keep the **subject line** concise — it becomes the changelog entry.
- The **type** determines the changelog section (`feat` → Added,
  `fix` → Fixed, `refactor`/`test`/`style` → Changed, `perf` →
  Performance, `docs` → Documentation, `build`/`chore`/`ci` →
  Maintenance).
- Include **PR references** as `(#N)` in the subject — cliff auto‑links
  them (e.g. `feat: add foo (#42)`).
- **Avoid headings** `#`–`###` in the body — they conflict with
  changelog structure (`##` = release, `###` = section). Use `####` if
  a heading is truly needed.
- Body text should be **plain prose or simple lists**. Numbered lists
  and sub‑items are fine but avoid deep nesting.

#### Breaking Changes

Breaking changes **must** use one of these conventional commit markers so
that `git‑cliff` can detect them and generate the
`### ⚠️ Breaking Changes` section in `CHANGELOG.md`:

- **Bang notation**: `feat!: remove deprecated API` (append `!` after the type/scope)
- **Footer trailer**: add `BREAKING CHANGE: <description>` as a
  [git trailer](https://git-scm.com/docs/git-interpret-trailers) at the
  end of the commit body

Examples of breaking changes: removing/renaming public API items, changing
default behaviour, bumping MSRV, altering serialisation formats.

---

## Validation Workflow

After modifying files, run appropriate validators.

Common commands:

```bash
just fix
just check
just ci
```

Refer to `docs/dev/commands.md` for full details.

---

## Testing Rules

Testing guidance lives in:

```text
docs/dev/testing.md
```

Key principle:

- Rust changes must pass unit tests, integration tests, and documentation builds.

---

## Project Context

- **Language**: Rust
- **Project**: d‑dimensional Delaunay triangulation library
- **MSRV**: 1.94
- **Edition**: 2024
- **Unsafe code**: forbidden (`#![forbid(unsafe_code)]`)

Architecture details are documented in:

```text
docs/code_organization.md
```

---

## Testing Execution Reference

Typical commands:

```bash
just test
just test-integration
just test-all
just examples
```

See `docs/dev/testing.md` for full testing guidance.

---

## Documentation Maintenance

- Never edit `CHANGELOG.md` manually
- Run `just changelog` to regenerate it from commits

---

## Agent Behavior Expectations

Agents should:

- Prefer small, focused patches
- Follow Rust idioms and borrowing conventions
- Avoid introducing allocations unless necessary
- Avoid panics in library code
- Search documentation under `docs/` when unsure

If multiple solutions exist, prefer the one that:

1. Preserves API stability
2. Maintains generic const‑dimension architecture
3. Keeps code simple and maintainable
