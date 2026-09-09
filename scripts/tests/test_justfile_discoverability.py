"""Regression tests for the public Just recipe surface."""

import json
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

import update_cargo_tool_pins
from subprocess_utils import run_safe_command

REPO_ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = REPO_ROOT / "justfile"
HELPER_JUSTFILE = REPO_ROOT / "just" / "helpers.just"
JUST_BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_just.sh"
JUST_VERSION_RESOLVER = REPO_ROOT / ".github" / "actions" / "setup-just" / "resolve-version.sh"
RECIPE_DECLARATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s+.*?)?:(?=\s|$)", re.MULTILINE)
WORKFLOW_VERSION_LOOKUP = re.compile(r"just --evaluate ([a-z0-9_]+_version)")
UNLOCKED_UV_RUN = re.compile(r"\buv\s+run\b(?!\s+--locked\b)")


@dataclass(frozen=True, kw_only=True)
class ZizmorAuthCase:
    """Synthetic credential sources and the token expected at the scanner boundary."""

    tokens: dict[str, str]
    expected_value: str
    gh_available: bool = True
    gh_stdout: str = ""
    gh_returncode: int = 0
    trace: bool = False


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
        timeout=30,
    )


def run_python_source_probe(tmp_path: Path, paths: list[str], *, git_returncode: int = 0) -> subprocess.CompletedProcess[str]:
    """Exercise the rendered recipe with isolated Git output and a tool-argument recorder."""
    git_output = tmp_path / "git-output.bin"
    git_output.write_bytes(b"".join(path.encode("utf-8") + b"\0" for path in paths))
    version = run_just("--evaluate", "uv_version").stdout.strip()
    rendered = run_just("--dry-run", "_python-tool", "ruff check")
    expected_git_args = "--no-pager ls-files --cached --others --exclude-standard --deduplicate -z -- *.py *.pyi"
    script = f"""
git() {{
    if [[ "$*" != {shlex.quote(expected_git_args)} ]]; then
        echo "Unexpected Git arguments: $*" >&2
        return 2
    fi
    cat {shlex.quote(git_output.as_posix())}
    return {git_returncode}
}}
uv() {{
    if [[ "$*" == "--version" ]]; then
        printf '%s\\n' {shlex.quote("uv " + version)}
    else
        printf '%s\\0' "$@"
    fi
}}
TMPDIR="$PWD"
{rendered.stdout}{rendered.stderr}
"""
    return run_safe_command("bash", ["-c", script], cwd=tmp_path, check=False, timeout=30)


def just_recipes() -> dict[str, dict[str, Any]]:
    """Return parsed recipe metadata from the pinned Just executable."""
    result = run_just("--dump", "--dump-format", "json")
    document = json.loads(result.stdout)
    recipes = document["recipes"]
    assert isinstance(recipes, dict)
    return recipes


def run_zizmor_probe(
    tmp_path: Path,
    case: ZizmorAuthCase,
    scan_returncode: int,
) -> subprocess.CompletedProcess[str]:
    """Exercise the rendered recipe with fake authentication and scanner commands."""
    version = run_just("--evaluate", "zizmor_version").stdout.strip()
    rendered = run_just("--dry-run", "zizmor")
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in case.tokens.items())
    script = f"""
unset ZIZMOR_GITHUB_TOKEN GH_TOKEN GITHUB_TOKEN GH_HOST ZIZMOR_OFFLINE ZIZMOR_NO_ONLINE_AUDITS
{exports}
command() {{
    if [[ "$*" == "-v gh" ]]; then
        return {0 if case.gh_available else 1}
    fi
    builtin command "$@"
}}
gh() {{
    printf '%s\\n' "$*" >> {shlex.quote((tmp_path / "gh.log").as_posix())}
    printf '%s\\n' {shlex.quote(case.gh_stdout)}
    echo "fixture-auth-diagnostic" >&2
    return {case.gh_returncode}
}}
zizmor() {{
    if [[ "$*" == "--version" ]]; then
        printf '%s\\n' {shlex.quote("zizmor " + version)}
        return 0
    fi
    if [[ "${{ZIZMOR_GITHUB_TOKEN:-}}" != {shlex.quote(case.expected_value)} ]]; then
        echo "Unexpected scanner credential" >&2
        return 97
    fi
    if [[ -n "${{GH_TOKEN:-}}" || -n "${{GITHUB_TOKEN:-}}" ]]; then
        echo "Conflicting scanner credential" >&2
        return 98
    fi
    printf 'scan:%s\\n' "$*"
    return {scan_returncode}
}}
{"set -x" if case.trace else ""}
{rendered.stdout}{rendered.stderr}
"""
    return run_safe_command("bash", ["-c", script], cwd=REPO_ROOT, check=False, timeout=30)


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


def test_local_and_ci_just_bootstrap_share_the_pinned_version_resolver() -> None:
    """Docs, local bootstrap, and CI should install the exact Justfile pin."""
    bash = shutil.which("bash")
    assert bash is not None
    resolved = subprocess.run(  # noqa: S603 - executable and script are repository-controlled.
        [bash, str(JUST_VERSION_RESOLVER)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()
    expected = run_just("--evaluate", "just_version").stdout.strip()
    action = (REPO_ROOT / ".github" / "actions" / "setup-just" / "action.yml").read_text(encoding="utf-8")
    bootstrap = JUST_BOOTSTRAP.read_text(encoding="utf-8")
    bootstrap_command = "bash scripts/bootstrap_just.sh"
    inline_install_command = 'cargo install --locked --version "$(bash .github/actions/setup-just/resolve-version.sh)" just'

    assert resolved == expected
    assert "bash .github/actions/setup-just/resolve-version.sh" in action
    assert "declaration_re=" not in action
    assert 'resolver="$repo_root/.github/actions/setup-just/resolve-version.sh"' in bootstrap
    assert 'pinned_version="$(bash "$resolver" "$repo_root/justfile")"' in bootstrap
    assert '[[ "$installed_version" == "just $pinned_version" ]]' in bootstrap
    assert 'cargo install --locked --version "$pinned_version" just' in bootstrap
    assert bootstrap_command in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert bootstrap_command in (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert inline_install_command not in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert inline_install_command not in (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for workflow_name in ("audit.yml", "benchmarks.yml", "papers.yml"):
        pull_request_paths, push_paths = workflow_trigger_paths(REPO_ROOT / ".github" / "workflows" / workflow_name)
        assert ".github/actions/setup-just/**" in pull_request_paths
        assert ".github/actions/setup-just/**" in push_paths


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


def test_ci_directly_lints_python_fixtures_with_full_ruff_policy() -> None:
    """CI must not drop fixture lint or replace configured rules with a subset."""
    recipes = just_recipes()
    dependencies = {dependency["recipe"] for dependency in recipes["ci"]["dependencies"]}
    result = run_just("--dry-run", "python-fixture-lint")
    commands = [shlex.split(line) for line in (result.stdout + result.stderr).splitlines() if line.startswith("uv run ")]

    assert "python-fixture-lint" in dependencies
    assert commands == [["uv", "run", "--locked", "ruff", "check", "tests/semgrep/"]]


def test_python_checks_and_fixer_share_source_discovery() -> None:
    """Every Python tool should consume the same file inventory."""
    recipes = just_recipes()
    expected_commands = {
        "python-format-check": ["ruff format --check"],
        "python-lint": ["ruff check"],
        "python-typecheck": ["--group notebooks ty check --error all"],
        "python-fix": ["ruff check --fix", "ruff format"],
    }
    for name, commands in expected_commands.items():
        result = run_just("--dry-run", name)
        rendered = result.stdout + result.stderr
        assert {dependency["recipe"] for dependency in recipes[name]["dependencies"]} == {"_python-tool"}
        for command in commands:
            assert f'uv run --locked {command} -- "${{python_files[@]}}"' in rendered


def test_python_source_discovery_preserves_paths_and_skips_deleted_files(tmp_path: Path) -> None:
    """Git-selected sources must reach the tool as paths, including leading dashes."""
    paths = ["scripts/owned.py", "tests/semgrep/scripts/tests/python_style.py", "new module.py", "new module.pyi", "--new.py"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Source discovery probe."""\n', encoding="utf-8")

    result = run_python_source_probe(tmp_path, [*paths, "deleted.py"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\0") == ["run", "--locked", "ruff", "check", "--", *paths, ""]
    assert not list(tmp_path.glob("delaunay-python-sources.*"))


def test_python_source_discovery_rejects_an_empty_file_set(tmp_path: Path) -> None:
    """Empty discovery must not let the tool fall back to an implicit root scan."""
    result = run_python_source_probe(tmp_path, [])

    assert result.returncode == 1
    assert result.stdout == ""
    assert "No Python source files found" in result.stderr
    assert not list(tmp_path.glob("delaunay-python-sources.*"))


def test_python_source_discovery_stops_after_partial_git_failure(tmp_path: Path) -> None:
    """A failing Git listing must stop before a partial source set reaches the tool."""
    (tmp_path / "partial.py").write_text('"""Partial listing probe."""\n', encoding="utf-8")
    result = run_python_source_probe(tmp_path, ["partial.py"], git_returncode=128)

    assert result.returncode == 128
    assert result.stdout == ""
    assert not list(tmp_path.glob("delaunay-python-sources.*"))


@pytest.mark.parametrize(
    "filename",
    [
        "scripts/typing_probe.py",
        "tests/semgrep/scripts/tests/python_exceptions.py",
        "tests/semgrep/scripts/tests/python_parse_boundaries.py",
        "tests/semgrep/scripts/tests/python_style.py",
    ],
)
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
    latest = run_just("--dry-run", "bench-latest")
    latest_command = latest.stdout + latest.stderr
    saved = run_just("--dry-run", "bench-save-baseline", "last")
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
    assert "[Performance Report](docs/PERFORMANCE.md)" in readme
    assert "legacy [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) report" in readme
    assert "provenance-limited release evidence" in readme
    assert "full report retains every benchmark and confidence interval" not in readme


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


def test_release_benchmark_summary_recipe_requires_strict_fresh_evidence() -> None:
    """The release summary recipe must propagate both freshness and strictness."""
    command = run_just("--dry-run", "bench-perf-summary")
    rendered = command.stdout + command.stderr

    assert "benchmark-utils generate-summary" in rendered
    assert "--run-benchmarks" in rendered
    assert "--profile perf" in rendered
    assert "--strict" in rendered


def test_local_and_sarif_semgrep_scans_share_target_enumeration() -> None:
    """Hosted uploads must scan the same tracked Python and Rust tests as local CI."""
    local = run_just("--dry-run", "semgrep")
    local_rendered = local.stdout + local.stderr
    sarif = run_just("--dry-run", "semgrep-scan", "semgrep-results.sarif")
    sarif_rendered = sarif.stdout + sarif.stderr
    workflow = (REPO_ROOT / ".github" / "workflows" / "semgrep-sarif.yml").read_text(encoding="utf-8")

    assert "scripts/semgrep_targets.py --null" in local_rendered
    assert "scripts/semgrep_targets.py --null" in sarif_rendered
    assert "--sarif --output" in sarif_rendered
    output_assignment = next(line for line in sarif_rendered.splitlines() if line.startswith("output="))
    assert shlex.split(output_assignment) == ["output=semgrep-results.sarif"]
    assert "just semgrep-scan semgrep-results.sarif" in workflow
    assert "git ls-files" not in workflow


def test_shared_semgrep_target_pathspecs_cover_both_test_languages_and_exclude_fixtures() -> None:
    """The target owner keeps ignored tests visible without scanning annotated violations."""
    target_source = (REPO_ROOT / "scripts" / "semgrep_targets.py").read_text(encoding="utf-8")

    assert '"scripts/tests/*.py"' in target_source
    assert '"tests/*.rs"' in target_source
    assert '":(exclude)tests/semgrep/**"' in target_source


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
    assert {dependency["recipe"] for dependency in recipes["_ensure-uv"]["dependencies"]} == {"_ensure-uv-available"}
    for name in ("_ensure-actionlint", "_ensure-shellcheck", "_ensure-shfmt", "_ensure-yamllint", "setup-tools"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-uv" in dependencies, name

    for name in ("_ensure-uv-stable", "update-dependencies", "update-python-dependencies"):
        dependencies = {dependency["recipe"] for dependency in recipes[name]["dependencies"]}
        assert "_ensure-uv-available" in dependencies, name
        assert "_ensure-uv" not in dependencies, name


@pytest.mark.parametrize("recipe", ["update", "update-cargo-tools"])
def test_update_preflights_stable_uv_before_mutations(recipe: str) -> None:
    """Reject unsupported uv output before dependency or installed-tool updates."""
    rendered_result = run_just("--dry-run", recipe)
    rendered = rendered_result.stdout + rendered_result.stderr
    preflight = "uv run --locked --no-sync --no-python-downloads python scripts/update_cargo_tool_pins.py --check-uv"

    assert rendered.count(preflight) == 1
    assert rendered.index("uv --version") < rendered.index(preflight)
    assert rendered.index(preflight) < rendered.index("cargo install-update --locked")
    if recipe == "update":
        assert rendered.index(preflight) < rendered.index("cargo upgrade --incompatible allow")
        assert rendered.index(preflight) < rendered.index("uv run --locked update-python-dev-pins")
    assert "installed_version=" not in rendered.split(preflight)[0]


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
    workflow_path = REPO_ROOT / ".github" / "workflows" / "papers.yml"
    pull_request_paths, push_paths = workflow_trigger_paths(workflow_path)
    required_paths = (
        ".python-version",
        ".github/actions/setup-just/**",
        "Cargo.lock",
        "Cargo.toml",
        "just/**",
        "rust-toolchain.toml",
        "scripts/notebook_validation_rendering.py",
        "scripts/subprocess_utils.py",
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
    recipes = just_recipes()
    ci_dependencies = [dependency["recipe"] for dependency in recipes["ci"]["dependencies"]]
    assert ci_dependencies[0] == "_validation-doc-figures-check-if-canonical"

    rendered_result = run_just("--dry-run", "validation-doc-figures-check")
    rendered = rendered_result.stdout + rendered_result.stderr
    assert 'check_root="target/docs/validation-figure-check"' in rendered
    assert 'generated_dir="target/notebooks/01_validation/validation_figures"' in rendered
    assert "DELAUNAY_VALIDATION_DOC_FIGURE_DIR" not in rendered
    assert "python -m notebook_validation_rendering" in rendered
    assert "docs/assets/validation" in rendered


def test_update_workflow_composes_scoped_dependency_and_tool_updates() -> None:
    """Update recipes should cover repo state without touching unrelated global tools."""
    recipes = just_recipes()
    update_dependencies = [dependency["recipe"] for dependency in recipes["update"]["dependencies"]]

    assert update_dependencies == ["_ensure-cargo-install-update", "_ensure-uv-stable", "update-dependencies", "update-cargo-tools"]

    aggregate_result = run_just("--dry-run", "update")
    aggregate_update = aggregate_result.stdout + aggregate_result.stderr
    assert aggregate_update.index("command -v cargo-install-update") < aggregate_update.index("cargo upgrade --incompatible allow")

    dependency_result = run_just("--dry-run", "update-dependencies")
    dependency_update = dependency_result.stdout + dependency_result.stderr
    dependency_preflights = [dependency["recipe"] for dependency in recipes["update-dependencies"]["dependencies"]]
    assert dependency_preflights[:2] == ["_ensure-cargo-edit", "_ensure-uv-available"]
    assert dependency_update.index("cargo_tool_has_exact_version") < dependency_update.index("cargo upgrade --incompatible allow")
    assert dependency_update.index("uv --version") < dependency_update.index("cargo upgrade --incompatible allow")
    assert "cargo upgrade --incompatible allow" in dependency_update
    fixture_manifest = "tests/fixtures/checkpoint_no_float_roundtrip/Cargo.toml"
    assert f"cargo upgrade --manifest-path {fixture_manifest} --incompatible allow" in dependency_update
    assert re.search(r"^cargo update$", dependency_update, re.MULTILINE) is not None
    assert f"cargo update --manifest-path {fixture_manifest}" in dependency_update
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
    assert "update-tool-pins" in tool_update
    assert "cargo install-update --all" not in tool_update
    assert "uv tool upgrade" not in tool_update
    package_block = re.search(r"packages=\(\n(?P<packages>.*?)\n\)", tool_update, re.DOTALL)
    assert package_block is not None
    updated_packages = set(re.findall(r"^\s+([a-z0-9-]+)$", package_block.group("packages"), re.MULTILINE))
    assert updated_packages == set(update_cargo_tool_pins.PIN_TO_PACKAGE.values())


def test_managed_tool_pins_exist_once_in_root_justfile() -> None:
    """Every managed Cargo package and uv should map to one root Just pin."""
    justfile_text = JUSTFILE.read_text(encoding="utf-8")

    for pin in update_cargo_tool_pins.PIN_TO_TOOL:
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


@pytest.mark.parametrize("scan_returncode", [0, 23])
@pytest.mark.parametrize(
    "case",
    [
        ZizmorAuthCase(
            tokens={"ZIZMOR_GITHUB_TOKEN": "fixture-zizmor", "GH_TOKEN": "fixture-gh", "GITHUB_TOKEN": "fixture-github"}, expected_value="fixture-zizmor"
        ),
        ZizmorAuthCase(tokens={"GH_TOKEN": "fixture-gh", "GITHUB_TOKEN": "fixture-github"}, expected_value="fixture-gh"),
        ZizmorAuthCase(tokens={"GITHUB_TOKEN": "fixture-github"}, expected_value="fixture-github"),
        ZizmorAuthCase(tokens={}, gh_stdout="fixture-auth", expected_value="fixture-auth"),
        ZizmorAuthCase(tokens={}, gh_stdout="fixture-partial-auth", gh_returncode=1, expected_value=""),
        ZizmorAuthCase(tokens={}, expected_value=""),
        ZizmorAuthCase(tokens={}, gh_available=False, expected_value=""),
        ZizmorAuthCase(tokens={"ZIZMOR_GITHUB_TOKEN": "fixture-traced"}, expected_value="fixture-traced", trace=True),
    ],
)
def test_zizmor_authentication_and_offline_fallback(
    tmp_path: Path,
    case: ZizmorAuthCase,
    scan_returncode: int,
) -> None:
    """Select credentials privately, report offline scans, and preserve scanner failures."""
    result = run_zizmor_probe(tmp_path, case, scan_returncode)

    assert result.returncode == scan_returncode, result.stderr
    expected_args = "--persona regular .github" if case.expected_value else "--offline --persona regular .github"
    assert result.stdout == f"scan:{expected_args}\n"
    assert ("online audits disabled" in result.stderr) == (not case.expected_value)
    assert "fixture-" not in result.stdout + result.stderr
    gh_log = tmp_path / "gh.log"
    if case.gh_available and not case.tokens:
        assert gh_log.read_text(encoding="utf-8") == "auth token --hostname github.com\n"
    else:
        assert not gh_log.exists()


def test_zizmor_sarif_workflow_uses_local_pin_and_online_persona(tmp_path: Path) -> None:
    """The hosted scanner must consume the evaluated local pin with online audits."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "zizmor.yml"
    workflow: Any = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)  # noqa: S506 - BaseLoader constructs data only.
    steps = workflow["jobs"]["analyze"]["steps"]
    setup = next(step for step in steps if step.get("uses") == "$/.github/actions/setup-just")
    resolver = next(step for step in steps if step.get("id") == "zizmor_version")
    scanners = [step for step in steps if step.get("uses", "").startswith("zizmorcore/zizmor-action@")]

    assert len(scanners) == 1
    scanner = scanners[0]
    assert steps.index(setup) < steps.index(resolver) < steps.index(scanner)
    assert scanner["with"]["version"] == "${{ steps.zizmor_version.outputs.version }}"
    assert scanner["with"]["online-audits"] == "true"
    assert scanner["with"]["persona"] == "regular"
    assert scanner["with"]["inputs"] == ".github"
    assert scanner["with"].get("advanced-security", "true") == "true"

    output_path = tmp_path / "github-output"
    script = f"export GITHUB_OUTPUT={shlex.quote(output_path.as_posix())}\n{resolver['run']}"
    run_safe_command("bash", ["-c", script], cwd=REPO_ROOT, timeout=30)
    expected_version = run_just("--evaluate", "zizmor_version").stdout.strip()
    assert output_path.read_text(encoding="utf-8") == f"version={expected_version}\n"
