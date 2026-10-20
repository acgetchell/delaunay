"""Consumer release policy using the installed shared target contract."""

import json
import os
import shlex
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from research_repo_tools.process import run_command
from research_repo_tools.release_assets import GitHubRelease, ReleaseTarget, parse_release_target

import benchmark_utils
import release_benchmarks

COMMIT = "a" * 40
TAG = "v0.8.2"
REPOSITORY = "acgetchell/delaunay"
ASSET = f"delaunay-{TAG}-criterion-baseline.tar.gz"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/release-benchmarks.yml"


@pytest.fixture
def target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReleaseTarget:
    """Provide the shared typed identity; remote lifecycle tests belong upstream."""
    selected = ReleaseTarget(repository=REPOSITORY, release_id=42, tag=TAG, commit=COMMIT)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setattr(release_benchmarks, "resolve_revision", lambda *_args: COMMIT)
    monkeypatch.setattr(release_benchmarks, "preflight_release_target", lambda *_args: selected)
    monkeypatch.setattr(
        release_benchmarks,
        "revalidate_release_target",
        lambda *_args: GitHubRelease(REPOSITORY, TAG, 42, draft=True, immutable=False, prerelease=False, assets=()),
    )
    return selected


def test_trusted_handoff_reaches_publication_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: ReleaseTarget) -> None:
    """The trusted target survives a long build and assets leave the draft mutable."""
    handoff = tmp_path / "target.json"
    publish = Mock()
    monkeypatch.setattr(release_benchmarks, "publish_release_asset", publish)
    release_benchmarks.release_benchmarks("preflight", tmp_path, handoff, REPOSITORY, TAG)
    assert parse_release_target(handoff.read_bytes()) == target
    assert (tmp_path / "output").read_bytes() == f"commit={COMMIT}\n".encode()
    publish.assert_not_called()
    release_benchmarks.release_benchmarks("publish", tmp_path, handoff, REPOSITORY, TAG)
    publish.assert_called_once_with(tmp_path, REPOSITORY, TAG, tmp_path / ASSET, target=target, publish=False)


def test_annotated_tag_handoff_uses_peeled_commit_and_requires_matching_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Git resolution distinguishes the tag object, its commit, and a later HEAD."""
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    root = tmp_path / "release checkout"
    root.mkdir()
    for args in (
        ["init", "--quiet"],
        ["config", "user.name", "Release Test"],
        ["config", "user.email", "release@example.invalid"],
        ["commit", "--quiet", "--allow-empty", "-m", "release"],
        ["tag", "--annotate", TAG, "--message", "release"],
    ):
        run_command("git", args, cwd=root, timeout=30)
    commit = run_command("git", ["--no-pager", "rev-parse", "HEAD"], cwd=root, timeout=30).stdout.strip()
    tag_object = run_command("git", ["--no-pager", "rev-parse", f"refs/tags/{TAG}"], cwd=root, timeout=30).stdout.strip()
    assert tag_object != commit
    run_command("git", ["commit", "--quiet", "--allow-empty", "-m", "later checkout"], cwd=root, timeout=30)
    selected = ReleaseTarget(repository=REPOSITORY, release_id=42, tag=TAG, commit=commit)
    preflight = Mock(return_value=selected)
    revalidate = Mock(return_value=GitHubRelease(REPOSITORY, TAG, 42, draft=True, immutable=False, prerelease=False, assets=()))
    publish = Mock()
    monkeypatch.setattr(release_benchmarks, "preflight_release_target", preflight)
    monkeypatch.setattr(release_benchmarks, "revalidate_release_target", revalidate)
    monkeypatch.setattr(release_benchmarks, "publish_release_asset", publish)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    handoff = tmp_path / "target.json"

    release_benchmarks.release_benchmarks("preflight", root, handoff, REPOSITORY, TAG)

    preflight.assert_called_once_with(root, REPOSITORY, TAG, commit)
    assert parse_release_target(handoff.read_bytes()) == selected
    assert output.read_bytes() == f"commit={commit}\n".encode()
    with pytest.raises(ValueError, match="HEAD differs"):
        release_benchmarks.release_benchmarks("publish", root, handoff, REPOSITORY, TAG)
    publish.assert_not_called()
    run_command("git", ["checkout", "--quiet", "--detach", commit], cwd=root, timeout=30)

    release_benchmarks.release_benchmarks("publish", root, handoff, REPOSITORY, TAG)

    publish.assert_called_once_with(root, REPOSITORY, TAG, root / ASSET, target=selected, publish=False)
    revalidate.assert_called_with(root, selected)


@pytest.mark.parametrize("failure", ["checkout", "tag", "repository", "existing"])
def test_consumer_policy_blocks_wrong_checkout_or_existing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: ReleaseTarget,
    failure: str,
) -> None:
    """Local policy adds checkout and no-overwrite constraints to shared lifecycle checks."""
    handoff = tmp_path / "target.json"
    release_benchmarks.release_benchmarks("preflight", tmp_path, handoff, REPOSITORY, TAG)
    original = handoff.read_bytes()
    publish = Mock()
    monkeypatch.setattr(release_benchmarks, "publish_release_asset", publish)
    if failure == "checkout":
        monkeypatch.setattr(release_benchmarks, "resolve_revision", lambda *_args: "b" * 40)
    if failure == "existing":
        monkeypatch.setattr(
            release_benchmarks,
            "revalidate_release_target",
            lambda *_args: GitHubRelease(REPOSITORY, TAG, target.release_id, draft=True, immutable=False, prerelease=False, assets=((ASSET, 7, 42, None),)),
        )
    with pytest.raises(ValueError, match=r"differs|already exists"):
        release_benchmarks.release_benchmarks(
            "publish",
            tmp_path,
            handoff,
            "different/repository" if failure == "repository" else REPOSITORY,
            "v0.8.3" if failure == "tag" else TAG,
        )
    publish.assert_not_called()
    assert handoff.read_bytes() == original


def test_release_workflow_preserves_trusted_code_runtime_and_target() -> None:
    """The release checkout cannot replace the upload implementation or its registry pin."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert workflow.index("Preserve publication runtime") < workflow.index("Validate draft release target") < workflow.index("Check out exact release commit")
    assert 'cp pyproject.toml uv.lock .python-version "$RUNNER_TEMP/release-tools/"' in workflow
    assert 'cp tooling/python/release_benchmarks.py "$RUNNER_TEMP/release-tools/"' in workflow
    assert workflow.count('"$RUNNER_TEMP/release-tools/.venv/bin/python"') == 2
    assert 'preflight "$RUNNER_TEMP/release-target.json"' in workflow
    assert 'publish "$RUNNER_TEMP/release-target.json"' in workflow


def test_release_workflow_staging_preserves_clean_source_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute workflow staging with real Git capture and archive creation, without sampling."""
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    checkout = tmp_path / "release checkout"
    checkout.mkdir()
    (checkout / ".gitignore").write_bytes((WORKFLOW.parents[2] / ".gitignore").read_bytes())
    for args in (
        ["init", "--quiet"],
        ["add", ".gitignore"],
        ["-c", "user.name=Release Test", "-c", "user.email=release@example.invalid", "commit", "--quiet", "-m", "release"],
    ):
        run_command("git", args, cwd=checkout, timeout=30)
    expected_source = benchmark_utils._source_state(checkout, version=TAG, ref=TAG)
    assert expected_source.git_clean is True

    executables = tmp_path / "bin"
    executables.mkdir()
    just = executables / "just"
    just.write_text('#!/bin/sh\nmkdir -p target/criterion\nprintf "raw samples\\n" > target/criterion/sample.json\n', encoding="utf-8")
    just.chmod(0o755)
    stub = tmp_path / "benchmark_boundary.py"
    stub.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(benchmark_utils.__file__).parent)!r})\n"
        "import benchmark_utils\n"
        "assert sys.argv[1:4] == ['run', '--locked', 'benchmark-utils']\n"
        "command = sys.argv[4]\n"
        "output = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "if command == 'generate-summary':\n"
        "    output.write_bytes(b'benchmark summary\\n')\n"
        "elif command == 'create-release-benchmark-metadata':\n"
        "    tag = sys.argv[sys.argv.index('--tag') + 1]\n"
        "    source = benchmark_utils._source_state(Path.cwd(), version=tag, ref=tag)\n"
        "    output.write_text(json.dumps(benchmark_utils._source_state_payload(source)), encoding='utf-8')\n"
        "else:\n"
        "    raise AssertionError(command)\n",
        encoding="utf-8",
    )
    uv = executables / "uv"
    uv.write_text(
        f'#!/bin/sh\nexec {shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(stub.as_posix())} "$@"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    output = tmp_path / "github-output"
    environment = os.environ | {
        "PATH": f"{executables}{os.pathsep}{os.environ.get('PATH', '')}",
        "GITHUB_OUTPUT": output.as_posix(),
        "RELEASE_TAG": TAG,
    }
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for name in ("Generate release benchmark summary", "Package release Criterion baseline"):
        command = next(step["run"] for step in workflow["jobs"]["release-baseline"]["steps"] if step.get("name") == name)
        run_command("bash", ["--noprofile", "--norc", "-eo", "pipefail", "-c", command], cwd=checkout, env=environment, timeout=30)

    assert output.read_text(encoding="utf-8") == f"asset={ASSET}\n"
    with tarfile.open(checkout / ASSET) as archive:
        members = {member.name.removeprefix("./"): member for member in archive.getmembers() if member.isfile()}
        assert set(members) == {"PERFORMANCE_RESULTS.md", "criterion/sample.json", "metadata.json"}
        contents: dict[str, bytes] = {}
        for name, member in members.items():
            extracted = archive.extractfile(member)
            assert extracted is not None
            with extracted:
                contents[name] = extracted.read()
    assert contents["PERFORMANCE_RESULTS.md"] == b"benchmark summary\n"
    assert contents["criterion/sample.json"] == b"raw samples\n"
    assert json.loads(contents["metadata.json"]) == benchmark_utils._source_state_payload(expected_source)


@pytest.mark.parametrize(
    ("step_name", "mode"),
    [("Validate draft release target", "preflight"), ("Attach and verify baseline on the draft", "publish")],
)
def test_release_workflow_launcher_ignores_incompatible_ambient_uv(tmp_path: Path, step_name: str, mode: str) -> None:
    """Both workflow commands reach the preserved adapter after ambient uv changes."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    command = next(step["run"] for step in workflow["jobs"]["release-baseline"]["steps"] if step.get("name") == step_name)
    runtime = tmp_path / "release-tools"
    python = runtime / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(Path(sys.executable).as_posix())} "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    adapter = runtime / "release_benchmarks.py"
    adapter.write_text("import json, sys\nprint(json.dumps([__file__, *sys.argv[1:]]))\n", encoding="utf-8")
    ambient_bin = tmp_path / "ambient-bin"
    ambient_bin.mkdir()
    uv = ambient_bin / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "incompatible uv version\\n" >&2\nprintf "called\\n" > "$RUNNER_TEMP/ambient-uv-called"\nexit 2\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    checkout = tmp_path / "release-checkout"
    checkout.mkdir()
    (checkout / "release_benchmarks.py").write_text('raise RuntimeError("release checkout adapter must not run")\n', encoding="utf-8")
    environment = os.environ.copy()
    environment["RUNNER_TEMP"] = tmp_path.as_posix()
    environment["PATH"] = f"{ambient_bin}{os.pathsep}{environment.get('PATH', '')}"
    result = run_command(
        "bash",
        ["--noprofile", "--norc", "-eo", "pipefail", "-c", command],
        cwd=checkout,
        env=environment,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual_adapter, actual_mode, actual_target = json.loads(result.stdout)
    assert Path(actual_adapter) == adapter
    assert actual_mode == mode
    assert Path(actual_target) == tmp_path / "release-target.json"
    assert not (tmp_path / "ambient-uv-called").exists()
