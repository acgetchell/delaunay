"""Tests for the installed Python utility package definition."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


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


def _build_support_wheel(repository: Path, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build the support wheel from a clean copied source tree."""
    configuration = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))
    source = tmp_path / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    for filename in ("LICENSE", "pyproject.toml", "uv.lock"):
        shutil.copy2(repository / filename, source / filename)
    shutil.copy2(repository / "scripts" / "README.md", scripts / "README.md")
    for module in configuration["tool"]["setuptools"]["py-modules"]:
        shutil.copy2(repository / "scripts" / f"{module}.py", scripts / f"{module}.py")

    uv = shutil.which("uv")
    assert uv is not None
    wheel_output = tmp_path / "dist"
    subprocess.run(  # noqa: S603 - executable is resolved; arguments are repository constants.
        [uv, "build", "--wheel", "--out-dir", str(wheel_output)],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(wheel_output.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0], configuration


def _venv_executable(venv: Path, name: str) -> Path:
    """Return a virtual-environment executable path on the current platform."""
    executable_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    return executable_dir / f"{name}{suffix}"


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


def test_support_package_uses_its_own_readme(tmp_path: Path) -> None:
    """Built Python metadata describes the support tools, not the Rust crate."""
    repository = Path(__file__).parents[2]
    wheel_path, configuration = _build_support_wheel(repository, tmp_path)

    assert configuration["project"]["readme"] == "scripts/README.md"
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        assert len(metadata_paths) == 1
        metadata = wheel.read(metadata_paths[0]).decode("utf-8").replace("\r\n", "\n")

    _headers, separator, description = metadata.partition("\n\n")
    assert separator
    assert description.startswith("# Scripts Directory\n")


def test_installed_notebook_extra_supports_default_lint_and_execute_modes(tmp_path: Path) -> None:
    """The built wheel's notebook extra installs every advertised command mode."""
    repository = Path(__file__).parents[2]
    wheel_path, configuration = _build_support_wheel(repository, tmp_path)
    assert set(configuration["project"]["optional-dependencies"]["notebooks"]) == {
        "ipykernel>=7.3.0",
        "nbclient>=0.11.0",
        "nbformat>=5.10.4",
        "ruff>=0.16.1",
        "ty>=0.0.66",
    }
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_path).decode("utf-8").replace("\r\n", "\n")
    assert "Provides-Extra: notebooks\n" in metadata
    for requirement in configuration["project"]["optional-dependencies"]["notebooks"]:
        assert f'Requires-Dist: {requirement}; extra == "notebooks"\n' in metadata

    uv = shutil.which("uv")
    assert uv is not None
    venv = tmp_path / "installed"
    subprocess.run(  # noqa: S603 - executable is resolved and arguments are test-owned paths.
        [uv, "venv", "--python", sys.executable, str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    python = _venv_executable(venv, "python")
    subprocess.run(  # noqa: S603 - executable is resolved and the wheel is a test-built local artifact.
        [uv, "pip", "install", "--no-cache", "--python", str(python), f"{wheel_path}[notebooks]"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    consumer = tmp_path / "consumer"
    consumer.mkdir()
    (consumer / "pyproject.toml").write_text('[project]\nname = "wheel-consumer"\nversion = "0.0.0"\nrequires-python = ">=3.14"\n', encoding="utf-8")
    notebook = consumer / "smoke.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "id": "installed-smoke",
                        "metadata": {},
                        "outputs": [],
                        "source": "value: int = 1\nprint(value)\n",
                    },
                ],
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
        ),
        encoding="utf-8",
    )
    notebook_check = _venv_executable(venv, "notebook-check")
    environment = os.environ.copy()
    environment["PATH"] = f"{notebook_check.parent}{os.pathsep}{environment.get('PATH', '')}"
    environment.pop("PYTHONPATH", None)

    lint = subprocess.run(  # noqa: S603 - executable and inputs come from the isolated test installation.
        [str(notebook_check), "lint", str(notebook), "--repo-root", str(consumer)],
        cwd=consumer,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    execute = subprocess.run(  # noqa: S603 - executable and inputs come from the isolated test installation.
        [str(notebook_check), "execute", str(notebook), "--repo-root", str(consumer), "--timeout", "30"],
        cwd=consumer,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert lint.returncode == 0, lint.stderr
    assert lint.stderr == ""
    assert "OK linted" in lint.stdout
    assert execute.returncode == 0, execute.stderr
    assert "Traceback" not in execute.stderr
    assert "OK executed" in execute.stdout
