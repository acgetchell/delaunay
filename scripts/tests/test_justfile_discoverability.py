"""Regression tests for the public Just recipe surface."""

import json
import re
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

import benchmark_utils
import update_cargo_tool_pins

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
HELPER_JUSTFILE = REPO_ROOT / "just" / "helpers.just"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)
WORKFLOW_VERSION_LOOKUP = re.compile(r"just --evaluate ([a-z0-9_]+_version)")
RELEASE_SIGNAL_TARGETS = re.compile(r"^\s*release-signal\)\s*\n\s*targets=\(([^)]*)\)", re.MULTILINE)
UNLOCKED_UV_RUN = re.compile(r"\buv\s+run\b(?!\s+--locked\b)")


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


def workflow_trigger_paths(path: Path) -> tuple[set[str], set[str]]:
    """Return pull-request and push path filters from one GitHub workflow."""
    workflow: Any = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506 - BaseLoader constructs data only.
    pull_request_paths = workflow["on"]["pull_request"]["paths"]
    push_paths = workflow["on"]["push"]["paths"]
    assert all(isinstance(item, str) for item in pull_request_paths)
    assert all(isinstance(item, str) for item in push_paths)
    return set(pull_request_paths), set(push_paths)


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


def test_run_recipe_uses_the_repository_lockfile() -> None:
    """The companion CLI should never resolve a different dependency graph."""
    result = run_just("--dry-run", "run")
    command = result.stdout + result.stderr

    assert "cargo run --locked --profile perf --features cli --bin delaunay --" in command


def test_cli_recipe_runs_binary_unit_and_integration_targets() -> None:
    """The maintained CLI lane should execute both feature-gated test targets."""
    result = run_just("--dry-run", "test-cli")
    command = result.stdout + result.stderr

    assert ("cargo nextest run --release --profile ci --features cli --bin delaunay --bin pachner-stress --test cli") in command


def test_check_code_includes_dependency_hygiene() -> None:
    """The comprehensive code check should include unused dependency analysis."""
    dependencies = {dependency["recipe"] for dependency in just_recipes()["check-code"]["dependencies"]}

    assert "unused-deps" in dependencies


def test_release_signal_benchmark_recipes_match_python_runner() -> None:
    """Just and Python should select the same curated release benchmark targets."""
    justfile_text = JUSTFILE.read_text(encoding="utf-8")
    suite_match = RELEASE_SIGNAL_TARGETS.search(justfile_text)
    assert suite_match is not None
    saved_baseline_targets = tuple(suite_match.group(1).split())

    latest = run_just("--dry-run", "bench-latest")
    latest_targets = tuple(re.findall(r"--bench ([A-Za-z0-9_-]+)", latest.stdout + latest.stderr))
    expected = benchmark_utils.RELEASE_SIGNAL_BENCH_TARGETS

    assert saved_baseline_targets == expected
    assert latest_targets == expected


def test_canonical_performance_recipes_share_the_cross_repository_contract() -> None:
    """Canonical release workflows should expose stable names and positional arguments."""
    recipes = just_recipes()
    assert {"performance-local", "performance-release", "performance-readme", "performance-doc", "performance-github-assets"} <= recipes.keys()
    assert {"perf-local", "perf-release", "perf-github-assets"}.isdisjoint(recipes)

    bench_parameters = recipes["bench-compare"]["parameters"]
    assert [parameter["name"] for parameter in bench_parameters] == ["baseline", "suite", "scope"]
    assert [parameter["default"] for parameter in bench_parameters] == ["last", "release-signal", "release-signal"]

    command = run_just("--dry-run", "bench-compare", "v0.7.8", "query", "all-benches")
    rendered = command.stdout + command.stderr
    assert 'bench-compare "v0.7.8" --suite "query" --scope "all-benches"' in rendered

    for name in ("performance-github-assets", "performance-release"):
        parameters = recipes[name]["parameters"]
        assert [parameter["name"] for parameter in parameters] == ["current_tag", "baseline_tag"]
        assert [parameter["default"] for parameter in parameters] == ["", ""]

        command = run_just("--dry-run", name, "v0.8.0", "v0.7.8")
        rendered = command.stdout + command.stderr
        assert f'benchmark-utils {name} "$current_tag" "$baseline_tag"' in rendered
        assert "current_tag='v0.8.0'" in rendered
        assert "baseline_tag='v0.7.8'" in rendered

    readme_command = run_just("--dry-run", "performance-readme")
    assert "uv run --locked publish-readme-performance" in readme_command.stdout + readme_command.stderr


def test_canonical_performance_recipes_shell_quote_tag_arguments() -> None:
    """Tag arguments must remain data in public recipes and their shared helper."""
    injected = 'v0.8.1"; printf injected; # '

    for recipe in ("performance-github-assets", "performance-release", "_performance-tag-pair-state"):
        command = run_just("--dry-run", recipe, injected, "v0.8.0")
        rendered = command.stdout + command.stderr
        current_assignment = next(line for line in rendered.splitlines() if line.startswith("current_tag="))
        baseline_assignment = next(line for line in rendered.splitlines() if line.startswith("baseline_tag="))

        assert shlex.split(current_assignment) == [f"current_tag={injected}"]
        assert shlex.split(baseline_assignment) == ["baseline_tag=v0.8.0"]


def test_release_metadata_recipe_uses_the_current_utc_date_internally() -> None:
    """Release preparation accepts only the target tag from the caller."""
    recipes = just_recipes()
    parameters = recipes["update-version"]["parameters"]

    assert [parameter["name"] for parameter in parameters] == ["tag"]
    command = run_just("--dry-run", "update-version", "v0.8.1")
    rendered = command.stdout + command.stderr
    assert "update-release-version 'v0.8.1'" in rendered
    assert "--release-date" not in rendered
    assert "check-docs-version-sync" in rendered
    assert "just update-version <tag>" in run_just().stdout


def test_release_workflows_fail_closed_before_writes_or_tag_mutation() -> None:
    """Release recipes should expose their non-mutating metadata gates."""
    recipes = just_recipes()
    strict_check = run_just("--dry-run", "release-version-check")
    assert "check-docs-version-sync --final-release" in strict_check.stdout + strict_check.stderr

    for name in ("tag", "tag-force"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "release-version-check" in dependencies

        injected = "v0.8.0; echo INJECTED"
        command = run_just("--dry-run", name, injected)
        rendered = command.stdout + command.stderr
        tag_command = next(line for line in rendered.splitlines() if line.startswith("uv run --locked tag-release "))
        expected = ["uv", "run", "--locked", "tag-release", injected]
        if name == "tag-force":
            expected.append("--force")
        assert shlex.split(tag_command) == expected

    changelog = run_just("--dry-run", "changelog-unreleased", "v0.8.1")
    rendered = changelog.stdout + changelog.stderr
    metadata_index = rendered.index("cargo metadata --locked --format-version 1 --no-deps")
    release_lookup_index = rendered.index('update-release-version "$version" --print-previous-release')
    cliff_index = rendered.index('git-cliff --tag "$version" -o CHANGELOG.md')
    assert metadata_index < release_lookup_index < cliff_index
    assert '[[ "$version" != "v$package_version" ]]' in rendered
    assert '--sync-changelog-date --previous-release "$previous_release"' in rendered


def test_canonical_performance_recipes_reject_partial_tag_pairs_before_dispatch() -> None:
    """A lone explicit tag must not reach the Python workflow command."""
    executable = shutil.which("just")
    assert executable is not None

    for recipe in ("performance-github-assets", "performance-release"):
        result = subprocess.run(  # noqa: S603 - executable is resolved and arguments are repository constants.
            [executable, recipe, "v0.8.0"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            encoding="utf-8",
        )

        assert result.returncode == 2
        assert "current_tag and baseline_tag must be provided together" in result.stderr
        assert "benchmark-utils" not in result.stdout + result.stderr


def test_cargo_tool_guards_reuse_pinned_helper() -> None:
    """Named Cargo-tool guards should share one exact-version implementation."""
    recipes = just_recipes()
    guard_names = (
        "_ensure-cargo-edit",
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
    for name in ("_ensure-actionlint", "_ensure-shellcheck", "_ensure-shfmt", "_ensure-yamllint", "setup-tools", "update-cargo-tools"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-uv" in dependencies, name


def test_setup_tools_closes_external_and_cargo_update_prerequisites() -> None:
    """Setup should fail early on gh, then provision and verify its update helper."""
    recipes = just_recipes()
    dependencies = [dependency["recipe"] for dependency in recipes["setup-tools"]["dependencies"]]
    body = json.dumps(recipes["setup-tools"]["body"])

    assert dependencies == ["_ensure-cargo", "_ensure-chktex", "_ensure-gh", "_ensure-jq", "_ensure-rustup", "_ensure-uv"]
    assert "External prerequisites that must already be on PATH: uv, gh, jq, rustup, cargo, and chktex." in body
    assert "unpinned cargo-update bootstrap helper" in body
    assert "cargo install --locked cargo-update" in body
    assert "cmds=(uv gh jq" in body
    assert "cmds+=(cargo-install-update" in body

    setup_result = run_just("--dry-run", "setup-tools")
    rendered = setup_result.stdout + setup_result.stderr
    first_mutation = rendered.index("uv sync --locked --group dev")
    for prerequisite in ("cargo", "chktex", "gh", "jq", "rustup"):
        assert rendered.index(f"command -v {prerequisite}") < first_mutation
    assert rendered.index("uv --version") < first_mutation


def test_validation_and_benchmark_uv_runs_are_locked() -> None:
    """Validation guards and benchmark workflows must reject lockfile drift."""
    paths = (
        HELPER_JUSTFILE,
        REPO_ROOT / ".github" / "workflows" / "benchmarks.yml",
        REPO_ROOT / ".github" / "workflows" / "generate-baseline.yml",
        REPO_ROOT / ".github" / "workflows" / "release-benchmarks.yml",
    )

    for path in paths:
        unlocked = UNLOCKED_UV_RUN.findall(path.read_text(encoding="utf-8"))
        assert unlocked == [], path


def test_performance_workflow_tracks_every_harness_input() -> None:
    """Performance checks should run when their code or toolchain changes."""
    pull_request_paths, push_paths = workflow_trigger_paths(REPO_ROOT / ".github" / "workflows" / "benchmarks.yml")
    required_paths = (
        ".python-version",
        "pyproject.toml",
        "rust-toolchain.toml",
        "scripts/benchmark_models.py",
        "scripts/performance_artifacts.py",
        "scripts/benchmark_utils.py",
        "scripts/hardware_utils.py",
        "scripts/subprocess_utils.py",
        "uv.lock",
    )

    for path in required_paths:
        assert path in pull_request_paths
        assert path in push_paths


def test_paper_workflow_tracks_validation_figure_producers() -> None:
    """Paper checks should run when Rust or notebook figure producers change."""
    pull_request_paths, push_paths = workflow_trigger_paths(REPO_ROOT / ".github" / "workflows" / "papers.yml")
    required_paths = (
        ".python-version",
        "Cargo.lock",
        "Cargo.toml",
        "just/**",
        "rust-toolchain.toml",
        "scripts/notebook_validation_rendering.py",
        "src/**",
    )

    for path in required_paths:
        assert path in pull_request_paths
        assert path in push_paths


def test_update_workflow_composes_scoped_dependency_and_tool_updates() -> None:
    """Update recipes should cover repo state without touching unrelated global tools."""
    recipes = just_recipes()
    update_dependencies = [dependency["recipe"] for dependency in recipes["update"]["dependencies"]]

    assert update_dependencies == ["_ensure-cargo-install-update", "update-dependencies", "update-cargo-tools"]

    aggregate_result = run_just("--dry-run", "update")
    aggregate_update = aggregate_result.stdout + aggregate_result.stderr
    assert aggregate_update.index("command -v cargo-install-update") < aggregate_update.index("cargo upgrade --incompatible allow")

    dependency_result = run_just("--dry-run", "update-dependencies")
    dependency_update = dependency_result.stdout + dependency_result.stderr
    dependency_preflights = [dependency["recipe"] for dependency in recipes["update-dependencies"]["dependencies"]]
    assert dependency_preflights[:2] == ["_ensure-cargo-edit", "_ensure-uv"]
    assert dependency_update.index("cargo_tool_has_exact_version") < dependency_update.index("cargo upgrade --incompatible allow")
    assert dependency_update.index("uv --version") < dependency_update.index("cargo upgrade --incompatible allow")
    assert "cargo upgrade --incompatible allow" in dependency_update
    assert "cargo update" in dependency_update
    assert "uv run --locked update-python-dev-pins" in dependency_update
    assert "uv lock --upgrade" in dependency_update
    assert dependency_update.index("uv run --locked update-python-dev-pins") < dependency_update.index("uv lock --upgrade")
    assert "uv sync --locked --group dev" in dependency_update
    assert "cargo install-update --all" not in dependency_update
    assert "uv tool upgrade" not in dependency_update

    tool_result = run_just("--dry-run", "update-cargo-tools")
    tool_update = tool_result.stdout + tool_result.stderr
    assert "command -v cargo-install-update" in tool_update
    assert "cargo install-update --locked" in tool_update
    assert "update-cargo-tool-pins" in tool_update
    assert "cargo install-update --all" not in tool_update
    assert "uv tool upgrade" not in tool_update
    package_block = re.search(r"packages=\(\n(?P<packages>.*?)\n\)", tool_update, re.DOTALL)
    assert package_block is not None
    updated_packages = set(re.findall(r"^\s+([a-z0-9-]+)$", package_block.group("packages"), re.MULTILINE))
    assert updated_packages == set(update_cargo_tool_pins.PIN_TO_PACKAGE.values())


def test_managed_cargo_tool_pins_exist_once_in_root_justfile() -> None:
    """Every managed Cargo package should map to one real root Just pin."""
    justfile_text = JUSTFILE.read_text(encoding="utf-8")

    for pin in update_cargo_tool_pins.PIN_TO_PACKAGE:
        assignments = re.findall(rf"^{re.escape(pin)}\s*:=", justfile_text, re.MULTILINE)
        assert len(assignments) == 1, pin


def test_workflow_tool_version_lookups_resolve_from_just() -> None:
    """GitHub Actions tool pins should resolve from the shared Just variables."""
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
    version_names = sorted(set(WORKFLOW_VERSION_LOOKUP.findall(workflow_text)))

    assert version_names
    for name in version_names:
        result = run_just("--evaluate", name)
        assert result.stdout.strip(), name
