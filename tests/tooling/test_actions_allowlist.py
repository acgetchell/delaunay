"""Exercise the installed strict checker with Delaunay's real policy and recipes."""

import json
import re
import shlex
import shutil
import sys
import tomllib
from pathlib import Path

import pytest
from research_repo_tools import action_updates
from research_repo_tools.cli import main
from research_repo_tools.process import run_command

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / ".github/settings/actions-selected.json"
AUTH = "rust-lang/crates-io-auth-action"
REVIEWED_COMMIT = "c6f97d42243bad5fab37ca0427f495c86d5b1a18"
SHA = "1" * 40


def check(root: Path, *inputs: str, policy: Path = POLICY) -> int:
    """Call the installed public CLI, retaining its real diagnostics and status."""
    return main(["--root", str(root), "actions", "allowlist", "--policy", str(policy), *inputs])


def test_all_checked_in_workflows_and_composites_pass() -> None:
    assert check(ROOT, ".github/workflows", ".github/actions") == 0


@pytest.mark.parametrize("ref", [REVIEWED_COMMIT, SHA, REVIEWED_COMMIT[:7], "main", "v1.0.5"])
@pytest.mark.parametrize("kind", ["step", "job", "composite"])
def test_authentication_commit_constraint(tmp_path: Path, capsys: pytest.CaptureFixture[str], ref: str, kind: str) -> None:
    templates = {
        "step": "jobs:\n  check:\n    steps:\n      - uses: {reference}\n",
        "job": "jobs:\n  check:\n    uses: {reference}\n",
        "composite": "runs:\n  using: composite\n  steps:\n    - uses: {reference}\n",
    }
    source = templates[kind].format(reference=f"{AUTH}@{ref}")
    path = tmp_path / "auth.yml"
    path.write_text(source, encoding="utf-8")
    assert check(tmp_path, path.name) == int(ref != REVIEWED_COMMIT)
    error = capsys.readouterr().err
    if ref != REVIEWED_COMMIT:
        line = next(i for i, text in enumerate(source.splitlines(), 1) if "uses:" in text)
        assert f"{path}:{line}:" in error
        assert f"permitted references: {AUTH}@{REVIEWED_COMMIT}" in error
    else:
        assert not error


@pytest.mark.parametrize(
    ("reference", "accepted"),
    [(f"github/codeql-action/{name}@{SHA}", True) for name in ("init", "analyze", "upload-sarif")]
    + [
        (f"github/codeql-action/unreviewed@{SHA}", False),
        (f"actions/cache/subpath@{SHA}", False),
        (f"actions/unreviewed@{SHA}", False),
        (f"github/unreviewed@{SHA}", False),
        (f"docker/unreviewed@{SHA}", False),
        (f"acgetchell/research-repo-tools/.github/workflows/dependabot-approve.yml@{SHA}", True),
        (f"acgetchell/research-repo-tools/.github/workflows/unreviewed.yml@{SHA}", False),
        ("./.github/actions/local", True),
        ("$/.github/actions/setup-just", True),
        ("docker://alpine:3", True),
    ],
)
def test_exact_identities_and_out_of_scope_references(tmp_path: Path, reference: str, *, accepted: bool) -> None:
    (tmp_path / "check.yml").write_text(f"jobs:\n  check:\n    steps:\n      - uses: {reference}\n", encoding="utf-8")
    assert check(tmp_path, "check.yml") == int(not accepted)


@pytest.mark.parametrize("mutation", ["github_owned_allowed", "verified_allowed", "actions/*@*", "github/*@*"])
def test_policy_cannot_enable_broad_exemptions(tmp_path: Path, capsys: pytest.CaptureFixture[str], mutation: str) -> None:
    policy = json.loads(POLICY.read_bytes())
    if mutation.endswith("_allowed"):
        policy[mutation] = True
    else:
        policy["patterns_allowed"].append(mutation)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert check(ROOT, ".github/workflows", policy=path) == 1
    assert str(path) in capsys.readouterr().err


def test_migrated_allowlist_fixture_preserves_syntax_and_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    """Keep the former regex's positive/negative cases at the parsed YAML boundary."""
    path = ROOT / "tests/semgrep/.github/workflows/action_policy.yml"
    expected = set()
    rejected = False
    for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "# allowlist: rejected" in text:
            rejected = True
        elif "uses:" in text:
            if rejected:
                expected.add(line)
            rejected = False
    assert len(expected) == 10
    assert check(ROOT, str(path)) == 1
    error = capsys.readouterr().err
    actual = {int(line) for line in re.findall(re.escape(str(path)) + r":(\d+):\d+ .*unapproved reference", error)}
    assert actual == expected, error


def test_real_recipe_rejects_new_unapproved_composite(tmp_path: Path) -> None:
    """Execute production Just scope, including new nested composite definitions."""
    shutil.copyfile(ROOT / "justfile", tmp_path / "justfile")
    shutil.copytree(ROOT / "just", tmp_path / "just")
    shutil.copytree(ROOT / ".github", tmp_path / ".github")
    command = ["--set", "rrt", f"{shlex.quote(sys.executable)} -m research_repo_tools", "action-allowlist"]
    assert run_command("just", command, cwd=tmp_path, check=False).returncode == 0
    action = tmp_path / ".github/actions/new/nested/action.yaml"
    action.parent.mkdir(parents=True)
    action.write_text(f"runs:\n  using: composite\n  steps:\n    - uses: actions/unreviewed@{SHA}\n", encoding="utf-8")
    result = run_command("just", command, cwd=tmp_path, check=False)
    assert result.returncode == 1
    assert f"{action}:4:" in result.stderr
    expanded = run_command("just", ["--dry-run", "ci"], cwd=ROOT)
    assert "actions allowlist --policy .github/settings/actions-selected.json .github/workflows .github/actions" in expanded.stderr
    assert "run: just ci" in (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_privileged_references_remain_excluded_from_updater() -> None:
    actions = tomllib.loads((ROOT / ".github/action-updates.toml").read_text(encoding="utf-8"))["actions"]
    assert AUTH not in actions
    assert not any("dependabot-approve" in name for name in actions)
    assert json.loads(POLICY.read_bytes())["patterns_allowed"].count(f"{AUTH}@{REVIEWED_COMMIT}") == 1


def test_codeql_update_preview_preserves_adopted_workflow_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The installed updater must resolve the consumer's selected tags without downgrades."""
    references: set[tuple[str, str, str]] = set()
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        references.update(re.findall(r"uses:\s+(github/codeql-action/[a-z-]+)@([0-9a-f]{40})\s+#\s*(v\d+\.\d+\.\d+)", workflow.read_text(encoding="utf-8")))
    assert {name for name, _sha, _version in references} == {f"github/codeql-action/{part}" for part in ("init", "analyze", "upload-sarif")}
    assert len(references) == 3
    adopted = {name: (sha, version) for name, sha, version in references}
    actions = tomllib.loads((ROOT / ".github/action-updates.toml").read_bytes().decode())["actions"]
    policy = tmp_path / "updates.toml"
    policy.write_text("[actions]\n" + "".join(f'"{name}" = "{actions[name]}"\n' for name in sorted(adopted)), encoding="utf-8")
    workflow = tmp_path / "workflow.yml"
    original = "jobs:\n  check:\n    steps:\n" + "".join(f"      - uses: {name}@{sha} # {version}\n" for name, (sha, version) in sorted(adopted.items()))
    workflow.write_text(original, encoding="utf-8", newline="\n")

    def resolve(action: str, target: str, _root: Path) -> tuple[str, str]:
        sha, version = adopted[action]
        return (sha if target == version else "1" * 40), target

    monkeypatch.setattr(action_updates, "resolve", resolve)
    assert main(["--root", str(tmp_path), "actions", "update", "--policy", str(policy), "--check", workflow.name]) == 0
    assert "Preview: 0 workflow files would change" in capsys.readouterr().out
    assert workflow.read_bytes() == original.encode()
