"""Tests for the installed Python utility package definition."""

import ast
import tomllib
from pathlib import Path


def _add_local_import(
    imported: set[str],
    qualified_name: str,
    local_modules: set[str],
    package_name: str,
) -> None:
    """Add the top-level local module named by *qualified_name*."""
    candidate = qualified_name.removeprefix(f"{package_name}.").split(".", 1)[0]
    if candidate in local_modules:
        imported.add(candidate)


def _local_imports(source: Path, local_modules: set[str]) -> set[str]:
    """Return repository-local modules imported by *source*."""
    imported: set[str] = set()
    package_name = source.parent.name
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
            if module == package_name:
                for alias in node.names:
                    _add_local_import(imported, alias.name, local_modules, package_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                _add_local_import(imported, alias.name, local_modules, package_name)
        if module is not None:
            _add_local_import(imported, module, local_modules, package_name)
    return imported


def test_local_imports_follows_package_level_imports(tmp_path: Path) -> None:
    """Package-level from-imports contribute their imported local modules."""
    package = tmp_path / "scripts"
    package.mkdir()
    source = package / "entry_point.py"
    source.write_text("from scripts import archive_changelog, external_module\n", encoding="utf-8")

    assert _local_imports(source, {"archive_changelog"}) == {"archive_changelog"}


def test_console_entry_point_import_closure_is_packaged() -> None:
    """Every local module reachable from a console entry point ships in the wheel."""
    repository = Path(__file__).parents[2]
    scripts_dir = repository / "scripts"
    configuration = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    packaged_modules = set(configuration["tool"]["setuptools"]["py-modules"])
    entry_modules = {target.partition(":")[0] for target in configuration["project"]["scripts"].values()}
    local_modules = {path.stem for path in scripts_dir.glob("*.py")}

    pending = list(entry_modules)
    reachable: set[str] = set()
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(_local_imports(scripts_dir / f"{module}.py", local_modules) - reachable)

    assert reachable <= packaged_modules
