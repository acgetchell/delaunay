"""Consumer release policy using the installed shared target contract."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from research_repo_tools.process import run_command
from research_repo_tools.release_assets import GitHubRelease, ReleaseTarget, parse_release_target

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
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/{TAG}")
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    monkeypatch.setattr(release_benchmarks, "resolve_revision", lambda *_args: COMMIT)
    monkeypatch.setattr(release_benchmarks, "preflight_release_target", lambda *_args, **_kwargs: selected)
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


@pytest.mark.parametrize("variable", ["GITHUB_EVENT_NAME", "GITHUB_REF", "GITHUB_SHA"])
def test_benchmark_preflight_rejects_another_dispatch_ref_or_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: ReleaseTarget, variable: str) -> None:
    """No handoff or measurement target is emitted for a main or mismatched dispatch."""
    monkeypatch.setenv(variable, "wrong")
    preflight = Mock(return_value=target)
    monkeypatch.setattr(release_benchmarks, "preflight_release_target", preflight)
    handoff = tmp_path / "target.json"
    with pytest.raises(ValueError, match="dispatch ref/source SHA"):
        release_benchmarks.release_benchmarks("preflight", tmp_path, handoff, REPOSITORY, TAG)
    preflight.assert_not_called()
    assert not handoff.exists()


def test_annotated_tag_handoff_uses_peeled_commit_and_requires_matching_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real Git resolution distinguishes the tag object, its commit, and a later HEAD."""
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/{TAG}")
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
    monkeypatch.setenv("GITHUB_SHA", commit)
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

    preflight.assert_called_once_with(root, REPOSITORY, TAG, commit, expected_title=TAG)
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


def test_release_workflow_separates_verification_from_writers() -> None:
    """Only the read-only verifier executes checkout code; writers use pinned RRT."""
    workflow = yaml.safe_load(WORKFLOW.read_bytes())
    assert set(workflow["jobs"]) == {"fetch", "verify", "attach"}
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {"tag", "candidate"}
    jobs = workflow["jobs"]
    for name in ("fetch", "attach"):
        job = jobs[name]
        assert job["permissions"]["contents"] == "write"
        assert not any(step.get("uses", "").startswith("actions/checkout@") for step in job["steps"])
        rendered = yaml.safe_dump(job)
        assert "python -I" in rendered
        assert "research-repo-tools==0.2.8" in rendered
        assert "setup-just" not in rendered
        assert "toolchain run" not in rendered
    assert jobs["verify"]["permissions"] == {"contents": "read", "checks": "read", "actions": "read"}
    assert jobs["verify"]["steps"][0]["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert jobs["attach"]["needs"] == "verify"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "performance-release" not in yaml.safe_dump(workflow)
    assert "cargo bench" not in yaml.safe_dump(workflow)


def test_release_workflow_binds_both_transport_attempts() -> None:
    """The original dispatch and both immutable artifact identities cross job boundaries."""
    jobs = yaml.safe_load(WORKFLOW.read_bytes())["jobs"]
    verify = yaml.safe_dump(jobs["verify"])
    attach = yaml.safe_dump(jobs["attach"])
    assert "needs.fetch.outputs.artifact-id" in verify
    assert "needs.fetch.outputs.artifact-digest" in verify
    assert "needs.fetch.outputs.validated-attempt" in verify
    assert "needs.verify.outputs.artifact-id" in attach
    assert "needs.verify.outputs.artifact-digest" in attach
    assert "needs.verify.outputs.validated-attempt" in attach
    assert "inputs.candidate" in attach
    assert "github.actor == 'acgetchell'" in jobs["fetch"]["if"]
    assert "github.triggering_actor == 'acgetchell'" in jobs["fetch"]["if"]
