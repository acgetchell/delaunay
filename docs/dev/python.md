# Python Development Guidelines

Guidance for repository-owned Python, including automation under `scripts/`
and static-analysis fixtures under `tests/semgrep/`.

The Rust library is the primary product, but the Python benchmark, changelog,
hardware, and release utilities are part of the trusted development workflow.
Keep them typed and predictable so failures are visible in CI instead of being
hidden behind loose mocks or broad exception handling.

---

## Validation

Run the Python validators through the repository toolchain:

```bash
just python-check
just test-python
```

`just python-check` runs Ruff formatting checks, Ruff linting, and
`just python-typecheck`. All three checks share a Git-derived list of tracked
Python files plus new, unignored `.py` and `.pyi` files, including files outside
`scripts/`. Deleted files are skipped. The fixer uses the same list.
`ty check --error all` is the type-checking authority. Prefer reducing untyped
surfaces in code and tests over adding more `ty` configuration.

`just python-fixture-lint` runs the full configured Ruff policy over
`tests/semgrep/` and is a direct dependency of `just ci`. Deliberate fixture
violations use exact per-file rule ignores or line-specific suppressions;
typing and annotation-import rules remain active. CodeRabbit excludes these
fixtures from general review and leaves Python docstring policy to Ruff.

`just check` also runs Python formatting checks, Ruff, and `ty` as part of the
normal repository validation bundle.

Jupyter notebook policy for authoring, cell metadata, execution, and artifacts
lives in [`notebooks.md`](notebooks.md). Notebook code still follows the Python
standards here, but notebook edits use the notebook validators rather than
treating `.ipynb` files as ordinary Python scripts.

---

## Typing

- Annotate all function and method parameters (including `*args` and `**kwargs`)
  and returns. Ruff enforces `ANN001`, `ANN002`, `ANN003`, `ANN201`, `ANN202`,
  `ANN204`, `ANN205`, and `ANN206`.
- Move annotation-only imports behind `if TYPE_CHECKING`; Ruff's configured
  `TC` rules, including `TC003`, apply to fixtures as well as scripts.
- On Python 3.14, use native deferred annotations and bare type names compatible
  with `UP037`. Do not add `from __future__ import annotations` solely for lint
  compliance. Document any runtime annotation consumer that requires strings.
- Prefer concrete standard-library types over `Any`, `dict`, or bare `Mock`
  when the shape is known.
- Keep helper signatures precise enough that `ty` can validate the call sites.
- Avoid growing type-checker configuration unless a demonstrated false positive
  cannot be solved cleanly in code.

---

## Subprocess Mocks

When mocking command wrappers such as `run_git_command()`,
`run_cargo_command()`, or `run_safe_command()`, prefer real typed subprocess
results:

```python
import subprocess


def completed_process(stdout: str = "", *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Return a typed subprocess result for command-wrapper mocks."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")
```

Use that helper instead of ad-hoc mocks such as:

```python
mock_result = Mock()
mock_result.stdout = "..."
mock_result.returncode = 0
```

Structured results make tests closer to production behavior and give `ty` real
attributes to check.

---

## Exceptions

- Catch specific recoverable error families in production code. Avoid
  `except Exception`.
- In tests, raise concrete exceptions that match the production recovery path
  (`OSError`, `RuntimeError`, `subprocess.CalledProcessError`,
  `subprocess.TimeoutExpired`, etc.).
- Do not use raw `Exception` in mocks just to force a fallback branch; doing so
  weakens the contract that the production code is meant to enforce.

---

## Test Helpers

Put reusable typed test helpers near the top of the test module or in
`scripts/tests/conftest.py` when they are shared. Prefer one helper that returns
the real structured type over repeating partially configured mocks throughout a
file.

## Parser and File-Format Contracts

When a script both writes and parses a text format, add a focused round-trip
test that writes representative records and parses them back. The test should
cover stable identifiers, optional sections, units, and numeric forms such as
scientific notation when those values can be emitted by production code.

For parser refactors, keep malformed-input regression tests for behavior that
callers depend on, such as skipping incomplete sections or failing loudly on
invalid numerical data.
