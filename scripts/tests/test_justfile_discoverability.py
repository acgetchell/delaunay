"""Regression tests for the public Just recipe surface."""

import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
HELPER_JUSTFILE = REPO_ROOT / "just" / "helpers.just"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)
WORKFLOW_VERSION_LOOKUP = re.compile(r"just --evaluate ([a-z0-9_]+_version)")


def run_just(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the repository's installed Just executable without a shell."""
    executable = shutil.which("just")
    assert executable is not None
    return subprocess.run(  # noqa: S603 - executable is resolved; arguments come from repository files.
        [executable, *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )


def just_recipes() -> dict[str, dict[str, Any]]:
    """Return parsed recipe metadata from the pinned Just executable."""
    result = run_just("--dump", "--dump-format", "json")
    document = json.loads(result.stdout)
    recipes = document["recipes"]
    assert isinstance(recipes, dict)
    return recipes


def test_recipe_declarations_are_lexicographically_sorted() -> None:
    """Recipe source order should support direct lookup by name."""
    for path in (JUSTFILE, HELPER_JUSTFILE):
        names = RECIPE_DECLARATION.findall(path.read_text(encoding="utf-8"))

        assert names == sorted(names), path


def test_bare_just_shows_curated_help() -> None:
    """Invoking Just without a recipe should never run a validation command."""
    result = run_just()

    assert result.stdout.startswith("Recommended workflows:\n")
    assert "Use 'just --list' for the complete grouped recipe reference." in result.stdout


def test_check_code_includes_dependency_hygiene() -> None:
    """The comprehensive code check should include unused dependency analysis."""
    dependencies = {dependency["recipe"] for dependency in just_recipes()["check-code"]["dependencies"]}

    assert "unused-deps" in dependencies


def test_cargo_tool_guards_reuse_pinned_helper() -> None:
    """Named Cargo-tool guards should share one exact-version implementation."""
    recipes = just_recipes()
    guard_names = (
        "_ensure-cargo-llvm-cov",
        "_ensure-cargo-machete",
        "_ensure-dprint",
        "_ensure-git-cliff",
        "_ensure-nextest",
        "_ensure-rumdl",
        "_ensure-samply",
        "_ensure-taplo",
        "_ensure-tectonic",
        "_ensure-tex-fmt",
        "_ensure-typos",
        "_ensure-zizmor",
    )

    for name in guard_names:
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-pinned-cargo-tool" in dependencies, name


def test_public_recipes_have_one_group_and_a_description() -> None:
    """Every listed recipe should explain its purpose in one stable section."""
    for name, recipe in just_recipes().items():
        if recipe["private"]:
            continue
        groups = [attribute["group"] for attribute in recipe["attributes"] if "group" in attribute]
        assert recipe["doc"], f"public recipe {name!r} has no description"
        assert len(groups) == 1, f"public recipe {name!r} has groups {groups!r}"


def test_public_recipes_do_not_duplicate_exact_behavior() -> None:
    """Public recipe names should not expose byte-for-byte duplicate implementations."""
    signatures: defaultdict[str, list[str]] = defaultdict(list)
    for name, recipe in just_recipes().items():
        if recipe["private"]:
            continue
        signature = json.dumps(
            {
                "body": recipe["body"],
                "dependencies": recipe["dependencies"],
                "parameters": recipe["parameters"],
            },
            sort_keys=True,
        )
        signatures[signature].append(name)

    duplicates = [names for names in signatures.values() if len(names) > 1]
    assert duplicates == []


def test_uv_backed_recipes_reuse_pinned_guard() -> None:
    """Local uv consumers should enforce the same pin consumed by workflows."""
    recipes = just_recipes()
    ensure_uv_body = json.dumps(recipes["_ensure-uv"]["body"])

    assert "uv --version" in ensure_uv_body
    assert "uv_version" in ensure_uv_body
    for name in ("_ensure-actionlint", "_ensure-shellcheck", "_ensure-shfmt", "_ensure-yamllint", "setup-tools"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-uv" in dependencies, name


def test_workflow_tool_version_lookups_resolve_from_just() -> None:
    """GitHub Actions tool pins should resolve from the shared Just variables."""
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
    version_names = sorted(set(WORKFLOW_VERSION_LOOKUP.findall(workflow_text)))

    assert version_names
    for name in version_names:
        result = run_just("--evaluate", name)
        assert result.stdout.strip(), name
