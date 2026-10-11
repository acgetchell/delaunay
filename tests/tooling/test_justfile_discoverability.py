"""Regression tests for the public Just recipe surface."""

import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from research_repo_tools.cargo_examples import discover_examples
from research_repo_tools.just_inspect import dry_run, inspect_justfile
from research_repo_tools.process import run_command as run_safe_command
from research_repo_tools.selection import select_files

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
HELPER_JUSTFILE = REPO_ROOT / "just" / "helpers.just"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)
UNLOCKED_UV_RUN = re.compile(r"\buv\s+run\b(?!\s+--locked\b)")


def run_pachner_stress_probe(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Capture literal artifact and Cargo arguments without running a stress workload."""
    rendered = dry_run(REPO_ROOT, "_pachner-stress-dim", args)
    script = f"""
uv() {{
    while [[ "$1" != "--" ]]; do shift; done
    shift
    "$@"
}}
cargo() {{
    printf '%s\\0' "$@" > {shlex.quote((tmp_path / "cargo-args").as_posix())}
}}
mkdir() {{
    printf '%s\\0' "$@" > {shlex.quote((tmp_path / "mkdir-args").as_posix())}
}}
{rendered.stdout}{rendered.stderr}
"""
    return run_safe_command("bash", ["-c", script], cwd=tmp_path, check=False, timeout=30)


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


def test_security_audit_covers_every_maintained_lockfile() -> None:
    """New isolated packages must not silently escape the dependency audit."""
    result = dry_run(REPO_ROOT, "security")
    commands = [shlex.split(line) for line in (result.stdout + result.stderr).splitlines() if line.startswith("uv run ")]
    audit, secrets = commands
    assert audit[:8] == ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "security", "osv"]
    maintained = select_files(REPO_ROOT, include=("uv.lock", "Cargo.lock", "**/Cargo.lock"))
    assert set(audit[8:]) == set(maintained)
    assert secrets[-2:] == ["security", "secrets"]


def test_bare_just_shows_curated_help() -> None:
    """Invoking Just without a recipe should never run a validation command."""
    result = run_safe_command("just", [], cwd=REPO_ROOT)

    assert result.stdout.startswith("Recommended workflows:\n")
    assert "Use 'just --list' for the complete grouped recipe reference." in result.stdout


def test_local_and_ci_setup_use_the_locked_package() -> None:
    """Local setup and CI consume the same pinned PyPI dependency group."""
    action = (REPO_ROOT / ".github" / "actions" / "setup-just" / "action.yml").read_text(encoding="utf-8")
    assert "version-file: pyproject.toml" in action
    assert "research-repo-tools toolchain sync" in action
    assert "research-repo-tools toolchain export" in action
    assert "cargo install" not in action
    for name in ("README.md", "CONTRIBUTING.md"):
        assert "uv run --locked --managed-python --only-group tooling research-repo-tools setup" in (REPO_ROOT / name).read_text(encoding="utf-8")
    for workflow_name in ("audit.yml", "benchmarks.yml", "papers.yml"):
        pull_request_paths, push_paths = workflow_trigger_paths(REPO_ROOT / ".github" / "workflows" / workflow_name)
        assert ".github/actions/setup-just/**" in pull_request_paths
        assert ".github/actions/setup-just/**" in push_paths


def test_run_recipe_uses_the_repository_lockfile() -> None:
    """The companion CLI should never resolve a different dependency graph."""
    result = dry_run(REPO_ROOT, "run")
    command = result.stdout + result.stderr

    assert "cargo run --locked --profile perf --features cli --bin delaunay --" in command


def test_cli_recipe_runs_binary_unit_and_integration_targets() -> None:
    """The maintained CLI lane should execute both feature-gated test targets."""
    result = dry_run(REPO_ROOT, "test-cli")
    command = result.stdout + result.stderr

    assert ("cargo nextest run --release --profile ci --features cli --bin delaunay --bin pachner-stress --test cli") in command


@pytest.mark.parametrize(
    "output_dir",
    ["artifacts with spaces", "artifacts/$trial", "artifacts/$(printf expanded)", "artifacts/`printf expanded`", "artifacts/single' double\""],
)
def test_pachner_stress_recipe_preserves_literal_output_paths(tmp_path: Path, output_dir: str) -> None:
    """Shell metacharacters in output paths must reach both file sinks unchanged."""
    result = run_pachner_stress_probe(tmp_path, ["3d", "5", "2", "1", output_dir, "round-trip"])

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "mkdir-args").read_text(encoding="utf-8").split("\0") == ["-p", output_dir, ""]
    assert (tmp_path / "cargo-args").read_text(encoding="utf-8").split("\0") == [
        "run",
        "--locked",
        "--profile",
        "perf",
        "--features",
        "cli",
        "--bin",
        "pachner-stress",
        "--",
        "--dimension",
        "3d",
        "--mode",
        "round-trip",
        "--vertices",
        "5",
        "--attempts",
        "2",
        "--validate-every",
        "1",
        "--progress-csv",
        f"{output_dir}/progress.csv",
        "--summary-json",
        f"{output_dir}/summary.json",
        "",
    ]


@pytest.mark.parametrize("argument_index", [1, 2, 3, 5])
def test_pachner_stress_recipe_rejects_literal_invalid_arguments(tmp_path: Path, argument_index: int) -> None:
    """Command substitutions must not turn invalid counts or modes into accepted values."""
    args = ["3d", "5", "2", "1", "artifacts", "round-trip"]
    args[argument_index] = f"$(printf {args[argument_index]})"
    result = run_pachner_stress_probe(tmp_path, args)

    assert result.returncode == 2, result.stderr
    assert args[argument_index] in result.stderr
    assert not (tmp_path / "mkdir-args").exists()
    assert not (tmp_path / "cargo-args").exists()


def test_check_code_includes_dependency_hygiene() -> None:
    """The comprehensive code check should include unused dependency analysis."""
    dependencies = {dependency["recipe"] for dependency in inspect_justfile(REPO_ROOT).recipes["check-code"]["dependencies"]}

    assert "unused-deps" in dependencies


def test_example_configuration_covers_cargo_targets_and_feature_builds() -> None:
    """New Cargo examples cannot silently escape the shared execution inventory."""
    examples = {example.name for example in discover_examples(REPO_ROOT, package="delaunay")}
    configuration = tomllib.loads((REPO_ROOT / "tooling/examples.toml").read_text(encoding="utf-8"))
    assert configuration["schema"] == 1
    assert configuration["package"] == "delaunay"
    assert configuration["profile"] == "release"
    assert configuration["build-timeout"] == 1800
    assert configuration["timeout"] == 600
    assert "include" not in configuration
    assert "exclude" not in configuration
    assert set(configuration["examples"]) == examples
    assert configuration["examples"]["diagnostics"]["features"] == ["diagnostics"]
    assert all(policy["expect"] for policy in configuration["examples"].values())
    rendered = dry_run(REPO_ROOT, "examples").stderr
    assert "toolchain run -- research-repo-tools validation cargo-examples tooling/examples.toml" in rendered
    assert "examples" in {item["recipe"] for item in inspect_justfile(REPO_ROOT).recipes["ci"]["dependencies"]}


def test_ci_directly_lints_python_fixtures_with_full_ruff_policy() -> None:
    """CI must not drop fixture lint or replace configured rules with a subset."""
    recipes = inspect_justfile(REPO_ROOT).recipes
    dependencies = {dependency["recipe"] for dependency in recipes["ci"]["dependencies"]}
    result = dry_run(REPO_ROOT, "python-fixture-lint")
    commands = [
        shlex.split(line) for line in (result.stdout + result.stderr).splitlines() if line.startswith("uv run ") and "research-repo-tools --version" not in line
    ]

    assert "python-fixture-lint" in dependencies
    assert commands == [["uv", "run", "--locked", "ruff", "check", "tests/semgrep/"]]


def test_python_checks_and_fixer_share_source_discovery() -> None:
    """Every Python tool should consume the same file inventory."""
    for name, action in (("python-check", "check"), ("python-fix", "fix"), ("python-typecheck", "typecheck")):
        result = dry_run(REPO_ROOT, name)
        assert f"research-repo-tools python {action}" in result.stdout + result.stderr
    for name in ("python-format-check", "python-lint"):
        result = dry_run(REPO_ROOT, name)
        assert "files run --include '*.py' --include '*.pyi'" in result.stdout + result.stderr


@pytest.mark.parametrize("filename", ["tooling/python/typing_probe.py", "tests/semgrep/tests/tooling/python_exceptions.py"])
def test_full_ruff_typing_policy_reaches_scripts_and_fixtures(filename: str) -> None:
    """Negative probes prove annotation and import guards remain blocking."""
    # Keep deliberately untyped probe text distinct from actual definitions so
    # Semgrep's source-level return-annotation regex does not mistake it for code.
    source_lines = [
        '"""Typing policy probe."""',
        "from pathlib import Path",
        "",
        "def missing_arguments(value, *args, **kwargs):",
        "    return value",
        "",
        "def _private():",
        "    return None",
        "",
        'def quoted(value: "Path") -> None:',
        "    pass",
        "",
        "class Example:",
        "    def __init__(self):",
        "        pass",
        "",
        "    @staticmethod",
        "    def static():",
        "        return None",
        "",
        "    @classmethod",
        "    def class_method(cls):",
        "        return None",
        "",
    ]
    source = "\n".join(source_lines)
    result = run_safe_command(
        sys.executable,
        ["-m", "ruff", "check", "--config", str(REPO_ROOT / "pyproject.toml"), "--output-format", "json", "--stdin-filename", filename, "-"],
        cwd=REPO_ROOT,
        input=source,
        check=False,
        timeout=30,
    )
    codes = {finding["code"] for finding in json.loads(result.stdout)}

    assert result.returncode == 1
    assert {"ANN001", "ANN002", "ANN003", "ANN201", "ANN202", "ANN204", "ANN205", "ANN206", "TC003", "UP037"} <= codes


def test_release_signal_benchmark_recipes_match_python_runner() -> None:
    """Just should delegate release measurements and baselines to the Python plan."""
    latest = dry_run(REPO_ROOT, "bench-latest")
    latest_command = latest.stdout + latest.stderr
    saved = dry_run(REPO_ROOT, "bench-save-baseline", ("last",))
    saved_command = saved.stdout + saved.stderr

    assert "uv run --locked benchmark-utils run-release-signal" in latest_command
    assert "cargo bench --profile perf --bench" not in latest_command
    assert 'uv run --locked benchmark-utils run-release-signal --save-baseline "$tag"' in saved_command


def test_checkpoint_baseline_replacement_and_full_report_are_discoverable() -> None:
    """Checkpoint payload identity and the active report should be unambiguous."""
    benchmark_docs = (REPO_ROOT / "benches" / "README.md").read_text(encoding="utf-8")
    normalized_benchmark_docs = re.sub(r"\s+", " ", benchmark_docs)
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "checkpoint-serialization/u32-payloads-v1" in normalized_benchmark_docs
    assert "row * 8 + column" in normalized_benchmark_docs
    assert "simplex vertex count" in normalized_benchmark_docs
    assert "unit-payload timings and saved Criterion baselines must not be compared" in normalized_benchmark_docs
    assert "[Performance Report][performance-report]" in readme
    assert "legacy [`docs/performance.md`][performance-report] report" in readme
    assert "[performance-report]: https://github.com/acgetchell/delaunay/blob/main/docs/performance.md" in readme
    assert "provenance-limited release evidence" in readme
    assert "full report retains every benchmark and confidence interval" not in readme


def test_canonical_performance_recipes_share_the_cross_repository_contract() -> None:
    """Canonical release workflows should expose stable names and positional arguments."""
    recipes = inspect_justfile(REPO_ROOT).recipes
    assert {"performance-local", "performance-release", "performance-readme", "performance-doc", "performance-github-assets"} <= recipes.keys()
    assert {"perf-local", "perf-release", "perf-github-assets"}.isdisjoint(recipes)

    bench_parameters = recipes["bench-compare"]["parameters"]
    assert [parameter["name"] for parameter in bench_parameters] == ["baseline", "suite", "scope"]
    assert [parameter["default"] for parameter in bench_parameters] == ["last", "release-signal", "release-signal"]

    command = dry_run(REPO_ROOT, "bench-compare", ("v0.7.8", "query", "all-benches"))
    rendered = command.stdout + command.stderr
    assert 'bench-compare "v0.7.8" --suite "query" --scope "all-benches"' in rendered

    for name in ("performance-github-assets", "performance-release"):
        parameters = recipes[name]["parameters"]
        assert [parameter["name"] for parameter in parameters] == ["current_tag", "baseline_tag"]
        assert [parameter["default"] for parameter in parameters] == ["", ""]

        command = dry_run(REPO_ROOT, name, ("v0.8.0", "v0.7.8"))
        rendered = command.stdout + command.stderr
        assert f'benchmark-utils {name} "$current_tag" "$baseline_tag"' in rendered
        assert "current_tag='v0.8.0'" in rendered
        assert "baseline_tag='v0.7.8'" in rendered

    readme_command = dry_run(REPO_ROOT, "performance-readme")
    assert "uv run --locked publish-readme-performance" in readme_command.stdout + readme_command.stderr


def test_canonical_performance_recipes_shell_quote_tag_arguments() -> None:
    """Tag arguments must remain data in public recipes and their shared helper."""
    injected = 'v0.8.1"; printf injected; # '

    for recipe in ("performance-github-assets", "performance-release", "_performance-tag-pair-state"):
        command = dry_run(REPO_ROOT, recipe, (injected, "v0.8.0"))
        rendered = command.stdout + command.stderr
        current_assignment = next(line for line in rendered.splitlines() if line.startswith("current_tag="))
        baseline_assignment = next(line for line in rendered.splitlines() if line.startswith("baseline_tag="))

        assert shlex.split(current_assignment) == [f"current_tag={injected}"]
        assert shlex.split(baseline_assignment) == ["baseline_tag=v0.8.0"]


def test_release_recipes_forward_shared_policy_arguments() -> None:
    """Release preparation exposes explicit dates and shared consistency gates."""
    recipes = inspect_justfile(REPO_ROOT).recipes
    assert [parameter["name"] for parameter in recipes["update-version"]["parameters"]] == ["tag", "args"]
    command = dry_run(REPO_ROOT, "update-version", ("v0.9.0", "--date", "2026-10-04", "--dry-run"))
    rendered = command.stdout + command.stderr
    assert 'research-repo-tools release update "$@"' in rendered
    assert "research-repo-tools release check" in rendered
    strict_check = dry_run(REPO_ROOT, "release-version-check")
    assert "research-repo-tools release check --final-release" in strict_check.stdout + strict_check.stderr
    for name in ("tag", "tag-force"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "release-version-check" in dependencies
        injected = "v0.8.0; echo INJECTED"
        command = dry_run(REPO_ROOT, name, (injected,))
        tag_command = next(line for line in (command.stdout + command.stderr).splitlines() if "research-repo-tools changelog tag " in line)
        expected = ["uv", "run", "--locked", "--group", "dev", "research-repo-tools", "changelog", "tag", injected]
        if name == "tag-force":
            expected.append("--force")
        assert shlex.split(tag_command) == expected
    command = dry_run(REPO_ROOT, "changelog-unreleased", ("v0.9.0", "2026-10-04"))
    assert "--tag 'v0.9.0' --date '2026-10-04'" in command.stdout + command.stderr


def test_release_benchmark_summary_recipe_requires_strict_fresh_evidence() -> None:
    """The release summary recipe must propagate both freshness and strictness."""
    command = dry_run(REPO_ROOT, "bench-perf-summary")
    rendered = command.stdout + command.stderr

    assert "benchmark-utils generate-summary" in rendered
    assert "--run-benchmarks" in rendered
    assert "--profile perf" in rendered
    assert "--strict" in rendered


def test_notebook_launch_and_reset_use_declared_shared_policies() -> None:
    """Launch provisions notebook backends; reset applies only through the explicit recipe."""
    launch = dry_run(REPO_ROOT, "notebook", ("--no-browser",))
    assert '--group dev --group notebooks research-repo-tools notebooks launch "$@"' in launch.stdout + launch.stderr
    reset = dry_run(REPO_ROOT, "notebook-reset-from-git", ("HEAD",))
    assert 'notebooks reset --revision "$source" --apply' in reset.stdout + reset.stderr
    assert "git restore" not in reset.stdout + reset.stderr


def test_local_and_sarif_semgrep_scans_share_target_enumeration() -> None:
    """Hosted uploads must scan the same tracked Python and Rust tests as local CI."""
    local = dry_run(REPO_ROOT, "semgrep")
    local_rendered = local.stdout + local.stderr
    sarif = dry_run(REPO_ROOT, "semgrep-scan", ("target/semgrep",))
    sarif_rendered = sarif.stdout + sarif.stderr
    workflow = (REPO_ROOT / ".github" / "workflows" / "semgrep-sarif.yml").read_text(encoding="utf-8")

    command = "semgrep scan --include '*.rs' --include 'tooling/python/*.py' --include 'tests/tooling/*.py'"
    assert command in local_rendered
    assert command in sarif_rendered
    assert "just semgrep-scan target/semgrep" in workflow
    assert "target/semgrep/semgrep.sarif" in workflow
    assert "git ls-files" not in workflow


def test_shared_semgrep_target_pathspecs_cover_both_test_languages_and_exclude_fixtures() -> None:
    """Repository pathspecs keep tests visible without scanning annotated violations."""
    targets = select_files(REPO_ROOT, include=("*.rs", "tooling/python/*.py", "tests/tooling/*.py", ".github/workflows/*.yml"), exclude=("tests/semgrep/**",))
    relative = set(targets)
    assert "tests/tooling/test_benchmark_utils.py" in relative
    assert "tests/proptest_sos.rs" in relative
    assert all(not path.startswith("tests/semgrep/") for path in relative)


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
    """Named tool guards delegate verification to the shared inventory."""
    recipes = inspect_justfile(REPO_ROOT).recipes
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
        assert dependencies == {"_ensure-toolchain"}, name


def test_public_recipes_have_one_group_and_a_description() -> None:
    """Every listed recipe should explain its purpose in one stable section."""
    for name, recipe in inspect_justfile(REPO_ROOT).recipes.items():
        if recipe["private"]:
            continue
        groups = [attribute["group"] for attribute in recipe["attributes"] if "group" in attribute]
        assert recipe["doc"], f"public recipe {name!r} has no description"
        assert len(groups) == 1, f"public recipe {name!r} has groups {groups!r}"


def test_public_recipes_do_not_duplicate_exact_behavior() -> None:
    """Public recipe names should not expose byte-for-byte duplicate implementations."""
    signatures: defaultdict[str, list[str]] = defaultdict(list)
    for name, recipe in inspect_justfile(REPO_ROOT).recipes.items():
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


def test_uv_backed_recipes_reuse_locked_guard() -> None:
    """Local uv consumers enforce the TOML pin through the installed package."""
    recipes = inspect_justfile(REPO_ROOT).recipes
    ensure_uv_body = json.dumps(recipes["_ensure-uv"]["body"])
    assert "uv run --locked --no-sync --no-python-downloads research-repo-tools --version" in ensure_uv_body
    for name in ("_ensure-actionlint", "_ensure-shellcheck", "_ensure-shfmt", "_ensure-yamllint"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-uv" in dependencies, name
    setup = json.dumps(recipes["setup-tools"]["body"])
    assert "research-repo-tools tectonic discover --format shell" in setup
    assert "uv run --locked --managed-python --only-group tooling research-repo-tools setup" in setup


def test_validation_and_benchmark_uv_runs_are_locked() -> None:
    """Validation guards and benchmark workflows must reject lockfile drift."""
    paths = (
        HELPER_JUSTFILE,
        REPO_ROOT / ".github" / "workflows" / "benchmarks.yml",
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
        "tooling/python/benchmark_models.py",
        "tooling/python/performance_artifacts.py",
        "tooling/python/benchmark_utils.py",
        "uv.lock",
    )

    for path in required_paths:
        assert path in pull_request_paths
        assert path in push_paths


def test_paper_workflow_tracks_validation_figure_producers() -> None:
    """Paper checks should run when Rust or notebook figure producers change."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "papers.yml"
    pull_request_paths, push_paths = workflow_trigger_paths(workflow_path)
    required_paths = (
        ".python-version",
        ".github/actions/setup-just/**",
        "Cargo.lock",
        "Cargo.toml",
        "just/**",
        "rust-toolchain.toml",
        "tooling/python/notebook_validation_rendering.py",
        "src/**",
    )

    for path in required_paths:
        assert path in pull_request_paths
        assert path in push_paths

    workflow = workflow_path.read_text(encoding="utf-8")
    assert "just validation-doc-figures-check" in workflow
    assert "git status --porcelain -- docs/assets/validation" not in workflow


def test_ci_composes_non_mutating_canonical_validation_figure_check() -> None:
    """The local CI contract should catch stale tracked figures on canonical macOS."""
    recipes = inspect_justfile(REPO_ROOT).recipes
    ci_dependencies = [dependency["recipe"] for dependency in recipes["ci"]["dependencies"]]
    assert ci_dependencies[0] == "_validation-doc-figures-check-if-canonical"

    rendered_result = dry_run(REPO_ROOT, "validation-doc-figures-check")
    rendered = rendered_result.stdout + rendered_result.stderr
    assert 'check_root="target/docs/validation-figure-check"' in rendered
    assert 'generated_dir="target/notebooks/01_validation/validation_figures"' in rendered
    assert "DELAUNAY_VALIDATION_DOC_FIGURE_DIR" not in rendered
    assert "python -m notebook_validation_rendering" in rendered
    assert "docs/assets/validation" in rendered


def test_workflows_share_managed_setup_and_online_sarif_policy() -> None:
    """Workflows inherit managed pins and keep hosted security scans online."""
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")))
    assert "just --evaluate" not in workflow_text
    assert "cargo install" not in workflow_text
    workflow_path = REPO_ROOT / ".github" / "workflows" / "zizmor.yml"
    workflow: Any = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506 - BaseLoader constructs data only.
    steps = workflow["jobs"]["analyze"]["steps"]
    setup = next(step for step in steps if step.get("uses") == "$/.github/actions/setup-just")
    scanner = next(step for step in steps if "research-repo-tools zizmor check" in step.get("run", ""))
    assert steps.index(setup) < steps.index(scanner)
    assert "--require-online --format sarif" in scanner["run"]
    assert scanner["env"]["GH_TOKEN"] == "${{ github.token }}"  # noqa: S105 - GitHub expression, never a credential.
    upload = next(step for step in steps if step.get("uses", "").startswith("github/codeql-action/upload-sarif@"))
    assert upload["with"]["sarif_file"] == "zizmor-results.sarif"
