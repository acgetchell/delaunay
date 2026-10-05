# Jupyter Notebook Guidelines

Notebook authoring, validation, execution, and artifact-ownership policy for
`notebooks/`.

The notebooks are reproducible front ends over repository APIs and binaries.
Keep reusable production, simulation, and data-processing logic in Rust or in a
typed support script; notebooks should orchestrate those components and render
or inspect their artifacts.

## Cell Identity And Source Hygiene

Every markdown, code, and raw cell must have a unique, stable, descriptive
`id`:

- use lowercase kebab-case, such as `load-validation-artifact` or
  `render-spherical-hero`
- name the cell's purpose rather than its position
- do not use random-looking IDs or generic names such as `cell-1`
- preserve an existing ID when editing a cell unless its purpose changes

`just notebook-check` enforces presence, uniqueness, and lowercase kebab-case.
Shared lint/advice owns notebook validation; `scripts/tests/test_notebook_policy.py`
enforces the consumer's spelling rule without a local orchestration script.
Stable IDs make notebook diffs, review comments, and nbformat validation easier
to follow.

Source notebooks must not commit generated outputs or execution counts. Keep
imports and deterministic repository-root discovery near the beginning, seed
random behavior that affects interpretation, and ensure cells run top to bottom
in a fresh kernel.

## Validation And Execution

Routine validation is lint-only:

```bash
just notebook-check
```

This command uses shared notebook structure/output checks, native Ruff and ty
notebook support, and advice, followed by the local ID spelling rule. It does
not extract cells into temporary Python files or execute them.

Consumer Ruff policy rejects `subprocess.Popen`, including aliases. Use
`subprocess.run(..., timeout=...)` in notebook cells so each process has a
bounded lifetime; reusable support scripts use the shared process runner.

The canonical macOS leg of `just ci` additionally runs
`just validation-doc-figures-check`. That named check executes only the
validation notebook, writes regenerated PNG files under `target/`, and compares
their exact bytes with the tracked figure set without publishing changes.
Linux and Windows retain the lint-only notebook check because canonical raster
bytes are platform-owned by the macOS paper workflow.

Execute one notebook deliberately when the task requires runtime validation:

```bash
just notebook-execute notebooks/00_quickstart.ipynb
```

The repository intentionally has no aggregate recipe that executes every
notebook. Notebook runtime depends on parameters, so notebook names and
directory layout must not encode a permanent `slow` category.

Exact command behavior and validator selection live in
[`commands.md`](commands.md).

## Generated And Tracked Artifacts

Shared execution writes `target/notebooks/notebooks/<notebook-stem>.ipynb`
and an adjacent `.report.json` with source/lock hashes and execution status.
Notebook-generated figures and data stay under
`target/notebooks/<notebook-stem>/`. Treat both as disposable scratch output;
the source notebook remains unchanged.

Tracked artifacts are refreshed only when the task explicitly includes the
artifact and through a named recipe. Current named workflows include:

- `just spherical-readme-hero` for
  `docs/assets/readme/delaunay_spherical_readme.png`
- `just validation-doc-figures` for validation figures under
  `docs/assets/validation/`
- `just validation-doc-figures-check` for a non-mutating currentness check of
  those figures on the canonical macOS platform

Canonical tracked figures belong under `docs/assets/`. Documentation and papers
should reference the same canonical asset instead of maintaining duplicate
copies. Artifact ownership and paper authorship boundaries live in
[`docs.md`](docs.md).

Do not direct an interactive validation-notebook run at the tracked asset
directory. `just validation-doc-figures` is the only refresh entry point: it
renders and validates the complete hierarchy-plus-five-level PNG set in staging,
then publishes the directory transactionally so a failed render, write, or
replacement preserves the previous complete set.

Use `just validation-doc-figures-check` when validating rather than refreshing.
It regenerates into `target/`, compares the complete canonical set, and reports
the exact stale names without changing tracked files.

Do not regenerate a potentially expensive tracked figure merely to test a
notebook. Use linting for routine validation and execute the relevant notebook
only when its runtime behavior or artifact is part of the task.

## Review Checklist

Before handing off a notebook change, confirm:

- every cell has a stable descriptive ID
- outputs and execution counts are cleared
- paths are repository-relative or derived from a discovered repository root
- randomness is deterministic when output interpretation depends on it
- subprocess calls use argument lists, timeouts, and actionable failure context
- generated scratch files stay under `target/`
- tracked artifacts use their documented named refresh recipe
- `just notebook-check` passes
