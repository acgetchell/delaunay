#!/usr/bin/env python3
"""Inspect, lint, and optionally execute Jupyter notebooks."""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, override

from subprocess_utils import run_safe_command

if TYPE_CHECKING:
    from collections.abc import Callable

RUFF_EXTEND_IGNORE = "INP001"
RUFF_LOCATION_RE = re.compile(r"\s*-->\s+.+?:(?P<line>\d+):(?P<column>\d+)")
RUFF_INLINE_LOCATION_RE = re.compile(r"^.+?:(?P<line>\d+):(?P<column>\d+):")
TY_LOCATION_RE = re.compile(r"^.+?:(?P<line>\d+):(?P<column>\d+): (?P<message>.+)$")
CELL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_CELL_ID_LENGTH = 64

type CellType = Literal["code", "markdown", "raw"]
VALID_CELL_TYPES: set[CellType] = {"code", "markdown", "raw"}


class NotebookExecutionError(RuntimeError):
    """Raised when optional notebook execution dependencies or runtime fail."""


@dataclass(frozen=True, slots=True)
class NotebookExecutionBackend:
    """Validated optional notebook execution imports."""

    client_factory: Callable[..., Any]
    read_notebook: Callable[..., Any]
    runtime_error_types: tuple[type[BaseException], ...]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Notebook lint diagnostic."""

    severity: str
    cell: int
    message: str


@dataclass(frozen=True, slots=True)
class CodeSnapshot:
    """Notebook code extracted into a Python-like source string."""

    source: str
    line_to_cell: dict[int, int]


@dataclass(frozen=True, slots=True)
class LintOptions:
    """Options that control notebook linting."""

    allow_outputs: bool = False
    strict: bool = False
    run_ruff: bool = True
    run_format: bool = True
    run_ty: bool = True
    project_root: Path | None = None


@dataclass(frozen=True, slots=True)
class NotebookCell:
    """Validated notebook cell metadata used by linting."""

    index: int
    cell_type: CellType
    cell_id: str | None
    source: str
    output_count: int
    execution_count: int | None

    @property
    def is_code(self) -> bool:
        """Return whether this is a Python code cell."""
        return self.cell_type == "code"


@dataclass(frozen=True, slots=True)
class NotebookDocument:
    """Validated notebook structure loaded from nbformat JSON."""

    path: Path
    nbformat: int
    nbformat_minor: int | None
    cells: tuple[NotebookCell, ...]


def parse_cell_type(value: Any, *, path: Path, index: int) -> CellType:
    """Parse a notebook cell type."""
    if not isinstance(value, str):
        msg = f"{path}: cell {index}: expected cell_type to be a string"
        raise TypeError(msg)
    if value not in VALID_CELL_TYPES:
        expected = ", ".join(sorted(VALID_CELL_TYPES))
        msg = f"{path}: cell {index}: expected cell_type to be one of {expected}; got {value!r}"
        raise ValueError(msg)
    return value


def parse_cell_id(value: Any, *, path: Path, index: int) -> str | None:
    """Parse an optional nbformat cell identifier."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    msg = f"{path}: cell {index}: id must be a string"
    raise TypeError(msg)


def parse_positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from error
    if parsed <= 0:
        msg = f"expected a positive integer, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def resolve_repo_root(path: Path) -> Path:
    """Resolve and validate the repository root path."""
    repo_root = path.resolve()
    if not repo_root.is_dir():
        msg = f"repository root does not exist or is not a directory: {repo_root}"
        raise FileNotFoundError(msg)
    return repo_root


def parse_cell_source(source: Any, *, path: Path, index: int) -> str:
    """Parse a notebook cell source as joined text."""
    if isinstance(source, list):
        if not all(isinstance(part, str) for part in source):
            msg = f"{path}: cell {index}: source list must contain only strings"
            raise TypeError(msg)
        return "".join(source)
    if isinstance(source, str):
        return source
    msg = f"{path}: cell {index}: source must be a string or list of strings, got {type(source).__name__}"
    raise TypeError(msg)


def parse_output_count(value: Any, *, path: Path, index: int) -> int:
    """Parse a notebook cell output list into its validated count."""
    if value is None:
        return 0
    if not isinstance(value, list):
        msg = f"{path}: cell {index}: outputs must be a list"
        raise TypeError(msg)
    return len(value)


def parse_execution_count(value: Any, *, path: Path, index: int) -> int | None:
    """Parse a notebook code-cell execution count."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    msg = f"{path}: cell {index}: execution_count must be an integer or null"
    raise TypeError(msg)


def parse_notebook_cell(raw_cell: Any, *, path: Path, index: int) -> NotebookCell:
    """Parse a raw nbformat cell into validated notebook metadata."""
    if not isinstance(raw_cell, dict):
        msg = f"{path}: cell {index}: expected cell to be an object"
        raise TypeError(msg)
    return NotebookCell(
        index=index,
        cell_type=parse_cell_type(raw_cell.get("cell_type"), path=path, index=index),
        cell_id=parse_cell_id(raw_cell.get("id"), path=path, index=index),
        source=parse_cell_source(raw_cell.get("source", ""), path=path, index=index),
        output_count=parse_output_count(raw_cell.get("outputs"), path=path, index=index),
        execution_count=parse_execution_count(raw_cell.get("execution_count"), path=path, index=index),
    )


def load_notebook(path: Path) -> NotebookDocument:
    """Load a notebook as plain JSON."""
    if not path.is_file():
        msg = f"notebook does not exist or is not a file: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    if not isinstance(notebook, dict):
        msg = f"{path}: expected notebook JSON to be an object"
        raise TypeError(msg)
    if notebook.get("nbformat") != 4:
        msg = f"{path}: expected nbformat 4, got {notebook.get('nbformat')!r}"
        raise ValueError(msg)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        msg = f"{path}: expected notebook cells to be a list"
        raise TypeError(msg)
    nbformat_minor = notebook.get("nbformat_minor")
    if nbformat_minor is not None and not isinstance(nbformat_minor, int):
        msg = f"{path}: expected nbformat_minor to be an integer or null"
        raise TypeError(msg)
    return NotebookDocument(
        path=path,
        nbformat=4,
        nbformat_minor=nbformat_minor,
        cells=tuple(parse_notebook_cell(cell, path=path, index=index) for index, cell in enumerate(cells, start=1)),
    )


def discover_notebooks(repo_root: Path) -> list[Path]:
    """Return notebooks under the conventional notebooks directory."""
    notebook_root = repo_root / "notebooks"
    if not notebook_root.is_dir():
        return []
    return sorted(path for path in notebook_root.glob("**/*.ipynb") if ".ipynb_checkpoints" not in path.parts)


def code_cells(notebook: NotebookDocument) -> list[NotebookCell]:
    """Return code cells with one-based cell numbers and joined source."""
    return [cell for cell in notebook.cells if cell.is_code]


def summarize(path: Path) -> None:
    """Print a compact notebook inventory."""
    notebook = load_notebook(path)
    print(f"{path}")
    print(f"  nbformat: {notebook.nbformat}.{notebook.nbformat_minor}")
    for cell in notebook.cells:
        first_line = next((line.strip() for line in cell.source.splitlines() if line.strip()), "")
        outputs = cell.output_count if cell.is_code else 0
        execution_count = cell.execution_count if cell.is_code else ""
        cell_id = cell.cell_id or "<missing>"
        print(
            f"  cell {cell.index:03d} {cell.cell_type:<8} "
            f"id={cell_id:<32} lines={len(cell.source.splitlines()):<3} "
            f"outputs={outputs:<2} exec={execution_count!s:<4} {first_line[:100]}",
        )


class NotebookVisitor(ast.NodeVisitor):
    """Collect notebook-specific Python quality diagnostics."""

    def __init__(self, cell: int) -> None:
        """Initialize a visitor for a one-based notebook cell number."""
        self.cell = cell
        self.diagnostics: list[Diagnostic] = []

    @override
    def visit_Import(self, node: ast.Import) -> None:
        """Warn on imports that are usually poor notebook defaults."""
        for alias in node.names:
            if alias.name == "pandas":
                self.diagnostics.append(Diagnostic("warning", self.cell, "imports pandas; prefer Polars unless pandas is required"))
            if alias.name == "csv":
                self.diagnostics.append(Diagnostic("warning", self.cell, "imports csv; prefer Polars for dataframe-shaped CSV analysis"))
        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Warn on from-imports that are usually poor notebook defaults."""
        if node.module == "pandas":
            self.diagnostics.append(Diagnostic("warning", self.cell, "imports pandas; prefer Polars unless pandas is required"))
        if node.module == "csv":
            self.diagnostics.append(Diagnostic("warning", self.cell, "imports csv; prefer Polars for dataframe-shaped CSV analysis"))
        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check regular function definitions for annotations."""
        self._check_function_annotations(node)
        self.generic_visit(node)

    @override
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check async function definitions for annotations."""
        self._check_function_annotations(node)
        self.generic_visit(node)

    @override
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Warn on broad exception handling in notebook code."""
        if node.type is None:
            self.diagnostics.append(Diagnostic("warning", self.cell, "uses bare except; catch specific exceptions"))
        elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            self.diagnostics.append(Diagnostic("warning", self.cell, f"catches broad {node.type.id}; catch specific recoverable errors"))
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """Check subprocess calls for shell and timeout hazards."""
        call_name = dotted_name(node.func)
        if call_name in {"subprocess.run", "subprocess.Popen"}:
            if keyword_bool(node, "shell"):
                self.diagnostics.append(Diagnostic("error", self.cell, f"{call_name} uses shell=True"))
            if call_name == "subprocess.run" and not has_keyword(node, "timeout"):
                self.diagnostics.append(Diagnostic("warning", self.cell, "subprocess.run lacks timeout; add one or document why it can run unbounded"))
            if call_name == "subprocess.Popen" and not has_wait_timeout(node):
                self.diagnostics.append(
                    Diagnostic("warning", self.cell, "subprocess.Popen stream lacks timeout; ensure tutorial commands cannot hang indefinitely"),
                )
        self.generic_visit(node)

    def _check_function_annotations(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        missing_args = [
            argument.arg
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if argument.arg not in {"self", "cls"} and argument.annotation is None
        ]
        if node.args.vararg is not None and node.args.vararg.annotation is None:
            missing_args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg is not None and node.args.kwarg.annotation is None:
            missing_args.append(f"**{node.args.kwarg.arg}")
        if missing_args:
            self.diagnostics.append(Diagnostic("warning", self.cell, f"function {node.name} lacks parameter annotations: {', '.join(missing_args)}"))
        if node.returns is None:
            self.diagnostics.append(Diagnostic("warning", self.cell, f"function {node.name} lacks return annotation"))


def dotted_name(node: ast.AST) -> str:
    """Return a dotted expression name when statically knowable."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def has_keyword(node: ast.Call, name: str) -> bool:
    """Return whether a call has a named keyword argument."""
    return any(keyword.arg == name for keyword in node.keywords)


def keyword_bool(node: ast.Call, name: str) -> bool:
    """Return a boolean keyword value when it is statically true."""
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is True
    return False


def has_wait_timeout(node: ast.Call) -> bool:
    """Return whether a Popen call obviously wraps a timeout in the same call."""
    return has_keyword(node, "timeout")


def extract_code(notebook: NotebookDocument) -> CodeSnapshot:
    """Extract code cells into one source string and retain line-to-cell mapping."""
    chunks: list[str] = []
    line_to_cell: dict[int, int] = {}
    current_line = 1
    cells = code_cells(notebook)
    for cell_position, cell in enumerate(cells):
        chunks.append(f"# %% notebook cell {cell.index}\n")
        line_to_cell[current_line] = cell.index
        current_line += 1
        source_lines = cell.source.splitlines(keepends=True)
        if not source_lines:
            chunks.append("\n")
            line_to_cell[current_line] = cell.index
            current_line += 1
        for source_line in source_lines:
            chunks.append(source_line)
            line_to_cell[current_line] = cell.index
            current_line += 1
        if source_lines and not source_lines[-1].endswith(("\n", "\r")):
            chunks.append("\n")
            line_to_cell[current_line] = cell.index
            current_line += 1
        if cell_position < len(cells) - 1:
            chunks.append("\n")
            line_to_cell[current_line] = cell.index
            current_line += 1
    return CodeSnapshot(source="".join(chunks), line_to_cell=line_to_cell)


def diagnostic_cell_from_ruff_line(line: str, snapshot: CodeSnapshot) -> int | None:
    """Return the notebook cell for either Ruff diagnostic location format."""
    match = RUFF_LOCATION_RE.match(line) or RUFF_INLINE_LOCATION_RE.match(line)
    if match is None:
        return None
    return snapshot.line_to_cell.get(int(match.group("line")), 0)


def ruff_lint_diagnostics(
    path: Path,
    notebook: NotebookDocument,
    project_root: Path,
) -> list[Diagnostic]:
    """Run Ruff lint checks on extracted notebook code when Ruff is available."""
    snapshot = extract_code(notebook)
    command = [
        "ruff",
        "check",
        "--stdin-filename",
        f"{path.stem}_notebook.py",
        "--extend-ignore",
        RUFF_EXTEND_IGNORE,
        "-",
    ]
    try:
        result = run_safe_command(
            command[0],
            command[1:],
            cwd=project_root,
            input=snapshot.source,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return [Diagnostic("error", 0, f"ruff timed out after {error.timeout} seconds")]

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return []
    if result.returncode != 1:
        return [Diagnostic("error", 0, f"ruff failed with exit code {result.returncode}:\n{output}")]

    diagnostics: list[Diagnostic] = []
    for block in output.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines or lines[0].startswith("Found "):
            continue
        cell = 0
        for line in lines:
            parsed_cell = diagnostic_cell_from_ruff_line(line, snapshot)
            if parsed_cell is not None:
                cell = parsed_cell
                break
        diagnostics.append(Diagnostic("error", cell, f"ruff check: {lines[0]}"))
    return diagnostics


def ruff_format_diagnostics(
    path: Path,
    notebook: NotebookDocument,
    project_root: Path,
) -> list[Diagnostic]:
    """Run Ruff format check on extracted notebook code when Ruff is available."""
    snapshot = extract_code(notebook)
    command = ["ruff", "format", "--check", "--stdin-filename", f"{path.stem}_notebook.py", "-"]
    try:
        result = run_safe_command(
            command[0],
            command[1:],
            cwd=project_root,
            input=snapshot.source,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return [Diagnostic("error", 0, f"ruff format timed out after {error.timeout} seconds")]

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode == 0:
        return []
    if result.returncode != 1:
        return [Diagnostic("error", 0, f"ruff format failed with exit code {result.returncode}:\n{output}")]
    return [Diagnostic("error", 0, f"ruff format: extracted notebook code is not formatted\n{output}")]


def ty_diagnostics(path: Path, notebook: NotebookDocument, project_root: Path) -> list[Diagnostic]:
    """Run ty on extracted notebook code when ty is available."""
    snapshot = extract_code(notebook)
    with tempfile.TemporaryDirectory(prefix="notebook-check-") as temporary_directory:
        extracted_path = Path(temporary_directory) / f"{path.stem}_notebook.py"
        extracted_path.write_text(snapshot.source, encoding="utf-8")
        command = [
            "ty",
            "check",
            "--project",
            str(project_root),
            "--output-format",
            "concise",
            str(extracted_path),
        ]
        try:
            result = run_safe_command(
                command[0],
                command[1:],
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return [Diagnostic("error", 0, f"ty timed out after {error.timeout} seconds")]

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return []
    if result.returncode not in {1, 2}:
        return [Diagnostic("error", 0, f"ty failed with exit code {result.returncode}:\n{output}")]

    diagnostics: list[Diagnostic] = []
    for line in output.splitlines():
        if not line.strip() or line.startswith("Found ") or line == "All checks passed!":
            continue
        match = TY_LOCATION_RE.match(line)
        if match is None:
            diagnostics.append(Diagnostic("error", 0, f"ty: {line}"))
            continue
        cell = snapshot.line_to_cell.get(int(match.group("line")), 0)
        diagnostics.append(Diagnostic("error", cell, f"ty: {match.group('message')}"))
    return diagnostics


def cell_id_diagnostics(notebook: NotebookDocument) -> list[Diagnostic]:
    """Return diagnostics for stable, unique notebook cell identifiers."""
    diagnostics: list[Diagnostic] = []
    first_cell_by_id: dict[str, int] = {}
    for cell in notebook.cells:
        cell_id = cell.cell_id
        if not cell_id:
            diagnostics.append(
                Diagnostic("error", cell.index, "missing cell id; add a stable descriptive lowercase kebab-case id"),
            )
            continue

        if len(cell_id) > MAX_CELL_ID_LENGTH or CELL_ID_RE.fullmatch(cell_id) is None:
            diagnostics.append(
                Diagnostic(
                    "error",
                    cell.index,
                    f"cell id {cell_id!r} must be 1-{MAX_CELL_ID_LENGTH} characters of lowercase kebab-case",
                ),
            )

        first_cell = first_cell_by_id.setdefault(cell_id, cell.index)
        if first_cell != cell.index:
            diagnostics.append(
                Diagnostic("error", cell.index, f"cell id {cell_id!r} duplicates cell {first_cell}; cell ids must be unique"),
            )
    return diagnostics


def code_cell_diagnostics(path: Path, notebook: NotebookDocument, options: LintOptions) -> list[Diagnostic]:
    """Return diagnostics from AST parsing and notebook output hygiene."""
    diagnostics: list[Diagnostic] = []
    for cell in code_cells(notebook):
        try:
            tree = ast.parse(cell.source, filename=f"{path}:cell-{cell.index}")
        except SyntaxError as error:
            diagnostics.append(Diagnostic("error", cell.index, f"syntax error: {error}"))
            continue
        visitor = NotebookVisitor(cell.index)
        visitor.visit(tree)
        diagnostics.extend(visitor.diagnostics)
        if cell.output_count > 0 and not options.allow_outputs:
            diagnostics.append(Diagnostic("error", cell.index, f"has {cell.output_count} output block(s); clear outputs before committing"))
        if cell.execution_count is not None and not options.allow_outputs:
            diagnostics.append(Diagnostic("error", cell.index, f"execution_count={cell.execution_count}; clear execution counts"))
    return diagnostics


def external_tool_diagnostics(path: Path, notebook: NotebookDocument, options: LintOptions) -> list[Diagnostic]:
    """Return diagnostics from Ruff and ty checks over extracted notebook code."""
    diagnostics: list[Diagnostic] = []
    project_root = options.project_root or Path.cwd()
    if options.run_ruff or options.run_format:
        if shutil.which("ruff") is None:
            diagnostics.append(Diagnostic("error", 0, "ruff is required for notebook linting; run through `uv run` or install Ruff"))
        else:
            if options.run_ruff:
                diagnostics.extend(ruff_lint_diagnostics(path, notebook, project_root))
            if options.run_format:
                diagnostics.extend(ruff_format_diagnostics(path, notebook, project_root))
    if options.run_ty:
        if shutil.which("ty") is None:
            diagnostics.append(Diagnostic("error", 0, "ty is required for notebook linting; run through `uv run` or install ty"))
        else:
            diagnostics.extend(ty_diagnostics(path, notebook, project_root))
    return diagnostics


def lint(path: Path, options: LintOptions) -> int:
    """Validate notebook JSON, compile code cells, and run Python lint checks."""
    notebook = load_notebook(path)
    diagnostics = [
        *cell_id_diagnostics(notebook),
        *code_cell_diagnostics(path, notebook, options),
        *external_tool_diagnostics(path, notebook, options),
    ]

    for diagnostic in diagnostics:
        stream = sys.stderr if diagnostic.severity == "error" else sys.stdout
        location = f"cell {diagnostic.cell}" if diagnostic.cell > 0 else "notebook"
        print(f"{path}: {location}: {diagnostic.severity}: {diagnostic.message}", file=stream)

    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return 1
    if options.strict and diagnostics:
        return 1
    return 0


def compact_exception(error: BaseException) -> str:
    """Render an exception as a single-line diagnostic fragment."""
    detail = " ".join(str(error).split())
    if not detail:
        return type(error).__name__
    return f"{type(error).__name__}: {detail}"


def nbclient_error_types(exceptions_module: Any) -> tuple[type[BaseException], ...]:
    """Return nbclient runtime error classes available in the installed version."""
    names = (
        "CellExecutionError",
        "CellTimeoutError",
        "DeadKernelError",
        "NotebookClientError",
    )
    error_types: list[type[BaseException]] = []
    for name in names:
        error_type = getattr(exceptions_module, name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(error_types)


def load_notebook_execution_backend(path: Path) -> NotebookExecutionBackend:
    """Load and validate optional notebook execution dependencies."""
    try:
        nbclient = import_module("nbclient")
        nbclient_exceptions = import_module("nbclient.exceptions")
        nbformat = import_module("nbformat")
    except ModuleNotFoundError as error:
        dependency = error.name or "notebook execution dependency"
        msg = f"{path}: execute(): missing optional dependency {dependency!r}; run through `uv run`"
        raise NotebookExecutionError(msg) from error

    client_factory = getattr(nbclient, "NotebookClient", None)
    if not callable(client_factory):
        msg = f"{path}: execute(): nbclient.NotebookClient is unavailable"
        raise NotebookExecutionError(msg)

    read_notebook = getattr(nbformat, "read", None)
    if not callable(read_notebook):
        msg = f"{path}: execute(): nbformat.read is unavailable"
        raise NotebookExecutionError(msg)

    runtime_error_types = nbclient_error_types(nbclient_exceptions)
    if not runtime_error_types:
        msg = f"{path}: execute(): nbclient runtime exception types are unavailable"
        raise NotebookExecutionError(msg)

    return NotebookExecutionBackend(
        client_factory=client_factory,
        read_notebook=read_notebook,
        runtime_error_types=runtime_error_types,
    )


def execute(path: Path, repo_root: Path, timeout: int) -> None:
    """Execute a notebook in memory without modifying it on disk."""
    backend = load_notebook_execution_backend(path)

    os.environ["MPLBACKEND"] = "Agg"
    with path.open(encoding="utf-8") as handle:
        notebook = backend.read_notebook(handle, as_version=4)
    try:
        client = backend.client_factory(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(repo_root)}},
        )
    except (AttributeError, TypeError, ValueError) as error:
        msg = f"{path}: execute(): NotebookClient creation failed: {compact_exception(error)}"
        raise NotebookExecutionError(msg) from error
    try:
        client.execute()
    except backend.runtime_error_types as error:
        msg = f"{path}: execute(): NotebookClient execution failed: {compact_exception(error)}"
        raise NotebookExecutionError(msg) from error
    print(f"OK executed {path}")


def lint_notebooks(paths: list[Path], options: LintOptions) -> int:
    """Lint every notebook in `paths`."""
    status = 0
    for path in paths:
        status = max(status, lint(path, options))
        if status == 0:
            print(f"OK linted {path}")
    return status


def execute_notebooks(paths: list[Path], repo_root: Path, timeout: int) -> None:
    """Execute every notebook in `paths`."""
    for path in paths:
        execute(path, repo_root, timeout)


def selected_notebooks(paths: list[Path], repo_root: Path) -> list[Path]:
    """Return explicit notebook paths or discover notebooks under the repo root."""
    if paths:
        return [path if path.is_absolute() else repo_root / path for path in paths]
    return discover_notebooks(repo_root)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, suggest_on_error=True, color=False)
    parser.add_argument("mode", choices=["summary", "lint", "execute"], help="notebook check mode")
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks to check; defaults to notebooks/**/*.ipynb")
    parser.add_argument("--allow-outputs", action="store_true", help="do not fail lint when code cells contain outputs or execution counts")
    parser.add_argument("--strict", action="store_true", help="treat warning diagnostics as lint failures")
    parser.add_argument("--no-ruff", action="store_true", help="skip Ruff lint checks for extracted notebook code")
    parser.add_argument("--no-format", action="store_true", help="skip Ruff format checks for extracted notebook code")
    parser.add_argument("--no-ty", action="store_true", help="skip ty checks for extracted notebook code")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="repository root for discovery and execution")
    parser.add_argument("--timeout", type=parse_positive_int, default=120, help="per-cell execution timeout in seconds")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Run notebook checker actions for parsed arguments."""
    repo_root = resolve_repo_root(args.repo_root)
    paths = selected_notebooks(args.notebooks, repo_root)
    if not paths:
        print("No notebooks found.")
        return 0
    if args.mode == "summary":
        for path in paths:
            summarize(path)
        return 0
    if args.mode == "lint":
        return lint_notebooks(
            paths,
            LintOptions(
                allow_outputs=args.allow_outputs,
                strict=args.strict,
                run_ruff=not args.no_ruff,
                run_format=not args.no_format,
                run_ty=not args.no_ty,
                project_root=repo_root,
            ),
        )
    execute_notebooks(paths, repo_root, args.timeout)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the requested notebook check."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        NotebookExecutionError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
