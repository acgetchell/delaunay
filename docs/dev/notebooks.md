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

This command validates notebook structure and metadata, extracts code cells,
and runs Ruff and ty without executing notebooks.

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

Ordinary notebook execution writes the executed notebook and generated files
under `target/notebooks/<notebook-stem>/`. Treat that directory as disposable
scratch output and leave the source notebook unchanged.

Tracked artifacts are refreshed only when the task explicitly includes the
artifact and through a named recipe. Current named workflows include:

- `just spherical-readme-hero` for
  `docs/assets/readme/delaunay_spherical_readme.png`
- `just validation-doc-figures` for validation figures under
  `docs/assets/validation/`

Canonical tracked figures belong under `docs/assets/`. Documentation and papers
should reference the same canonical asset instead of maintaining duplicate
copies. Artifact ownership and paper authorship boundaries live in
[`docs.md`](docs.md).

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
