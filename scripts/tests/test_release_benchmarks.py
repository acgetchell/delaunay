"""Execute the release publication guard against a local fake GitHub transport."""

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from subprocess_utils import run_safe_command

if TYPE_CHECKING:
    import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "release_benchmarks.sh"
COMMIT = "a" * 40
ASSET = "delaunay-v0.8.2-criterion-baseline.tar.gz"


@pytest.fixture
def release_env(tmp_path: Path) -> dict[str, str]:
    """Provide a draft, tagged commit, archive, and observable fake CLI writes."""
    archive = b"complete benchmark archive"
    (tmp_path / ASSET).write_bytes(archive)
    release = {
        "id": 42,
        "tag_name": "v0.8.2",
        "draft": True,
        "prerelease": False,
        "immutable": False,
        "published_at": None,
        "assets": [],
    }
    (tmp_path / "before.json").write_text(json.dumps(release), encoding="utf-8")
    release["assets"] = [{"name": ASSET, "state": "uploaded", "size": len(archive), "digest": f"sha256:{hashlib.sha256(archive).hexdigest()}"}]
    (tmp_path / "after.json").write_text(json.dumps(release), encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh = fake_bin / "gh"
    gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_ROOT/calls"
case "$*" in
    *'--method PATCH'*) exit "${PUBLISH_STATUS:-0}" ;;
    'release upload '*)
        [[ "${UPLOAD_STATUS:-0}" == 0 ]] || exit "$UPLOAD_STATUS"
        cp "$FAKE_ROOT/after.json" "$FAKE_ROOT/before.json" ;;
    *'/releases?per_page=100'*)
        [[ "${MISSING_RELEASE:-0}" == 0 ]] || exit 1
        if [[ -f "$FAKE_ROOT/releases-response.json" ]]; then
            cat "$FAKE_ROOT/releases-response.json"
        else
            jq '[[.]]' "$FAKE_ROOT/before.json"
        fi
        exit "${RELEASES_STATUS:-0}" ;;
    *'/git/ref/tags/'*) printf '{"type":"commit","sha":"%s"}\\n' "$REMOTE_COMMIT" ;;
    *) exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    git = fake_bin / "git"
    git.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$LOCAL_COMMIT"\n', encoding="utf-8")
    gh.chmod(0o755)
    git.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_ROOT": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_REPOSITORY": "owner/repo",
        "RELEASE_TAG": "v0.8.2",
        "LOCAL_COMMIT": COMMIT,
        "REMOTE_COMMIT": COMMIT,
        "EXPECTED_COMMIT": COMMIT,
        "EXPECTED_RELEASE_ID": "42",
    }


def run_guard(tmp_path: Path, env: dict[str, str], mode: str = "preflight") -> subprocess.CompletedProcess[str]:
    """Run the actual guard without remote access or Git mutations."""
    return run_safe_command("bash", [str(SCRIPT), mode], cwd=tmp_path, env=env, check=False, timeout=10)


def change_release(tmp_path: Path, field: str, value: object, filename: str = "before.json") -> None:
    """Change one transport field to exercise a fail-closed boundary."""
    path = tmp_path / filename
    release = json.loads(path.read_text(encoding="utf-8"))
    release[field] = value
    path.write_text(json.dumps(release), encoding="utf-8")


def assert_no_publication(tmp_path: Path) -> None:
    """Check that no publication request escaped a failed validation."""
    calls = tmp_path / "calls"
    assert not calls.exists() or "--method PATCH" not in calls.read_text(encoding="utf-8")


def assert_no_remote_writes(tmp_path: Path) -> None:
    """Require failed pre-upload validation to leave the remote draft untouched."""
    assert_no_publication(tmp_path)
    calls = tmp_path / "calls"
    assert not calls.exists() or "release upload" not in calls.read_text(encoding="utf-8")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("tag", ["main", "v01.2.3", "v1.2.3-rc.1", "v1.2.3+build", "v1.2.3\n", "$(touch injected)"])
def test_invalid_tag_fails_before_network(tmp_path: Path, release_env: dict[str, str], tag: str) -> None:
    release_env["RELEASE_TAG"] = tag
    assert run_guard(tmp_path, release_env).returncode != 0
    assert not (tmp_path / "calls").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("draft", False), ("prerelease", True), ("immutable", True), ("immutable", None), ("published_at", "2026-09-09"), ("tag_name", "v0.8.1"), ("id", 43)],
)
def test_unsuitable_release_never_publishes(tmp_path: Path, release_env: dict[str, str], field: str, value: object) -> None:
    change_release(tmp_path, field, value)
    assert run_guard(tmp_path, release_env).returncode != 0
    assert_no_publication(tmp_path)


@pytest.mark.parametrize("failure", ["missing", "moved", "existing"])
def test_preflight_rejects_missing_or_changed_target(tmp_path: Path, release_env: dict[str, str], failure: str) -> None:
    if failure == "missing":
        release_env["MISSING_RELEASE"] = "1"
    elif failure == "moved":
        release_env["REMOTE_COMMIT"] = "b" * 40
    else:
        change_release(tmp_path, "assets", [{"name": ASSET}])
    assert run_guard(tmp_path, release_env).returncode != 0
    assert_no_publication(tmp_path)


def test_preflight_binds_release_and_commit_without_writes(tmp_path: Path, release_env: dict[str, str]) -> None:
    result = run_guard(tmp_path, release_env)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text(encoding="utf-8") == f"release_id=42\ncommit={COMMIT}\n"
    assert "release upload" not in (tmp_path / "calls").read_text(encoding="utf-8")
    assert_no_publication(tmp_path)


def test_publish_verifies_upload_before_publication(tmp_path: Path, release_env: dict[str, str]) -> None:
    result = run_guard(tmp_path, release_env, "publish")
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert calls.index("release upload") < calls.rindex("/releases?per_page=100") < calls.index("--method PATCH")
    assert "-F draft=false" in calls
    assert "--clobber" not in calls


@pytest.mark.parametrize("failure", ["upload", "digest", "absent", "published", "identity", "empty", "missing_identity"])
def test_failed_publication_leaves_draft_recoverable(tmp_path: Path, release_env: dict[str, str], failure: str) -> None:
    if failure == "upload":
        release_env["UPLOAD_STATUS"] = "1"
    elif failure in {"digest", "absent"}:
        assets = [] if failure == "absent" else [{"name": ASSET, "state": "uploaded", "size": 1, "digest": "sha256:wrong"}]
        change_release(tmp_path, "assets", assets, "after.json")
    elif failure == "published":
        change_release(tmp_path, "draft", value=False, filename="after.json")
    elif failure == "identity":
        change_release(tmp_path, "id", 43, "after.json")
    elif failure == "empty":
        (tmp_path / ASSET).write_bytes(b"")
    else:
        release_env.pop("EXPECTED_COMMIT")
    assert run_guard(tmp_path, release_env, "publish").returncode != 0
    assert_no_publication(tmp_path)


@pytest.mark.parametrize("mode", ["preflight", "publish"])
@pytest.mark.parametrize("response", ["", "null", "[]", "[[]]", "[null]", "{"])
def test_malformed_release_response_never_writes(tmp_path: Path, release_env: dict[str, str], response: str, mode: str) -> None:
    (tmp_path / "releases-response.json").write_text(response, encoding="utf-8")
    assert run_guard(tmp_path, release_env, mode).returncode != 0
    assert_no_remote_writes(tmp_path)


@pytest.mark.parametrize("mode", ["preflight", "publish"])
def test_release_response_followed_by_transport_failure_never_writes(tmp_path: Path, release_env: dict[str, str], mode: str) -> None:
    release_env["RELEASES_STATUS"] = "57"
    assert run_guard(tmp_path, release_env, mode).returncode == 57
    assert_no_remote_writes(tmp_path)


@pytest.mark.parametrize("filename", ["before.json", "after.json"])
@pytest.mark.parametrize("assets", [None, {}, {"asset": {"name": "other"}}, False, 0, "assets", [None], [{}], [{"name": None}], [{"name": 3}], [{"name": ""}]])
def test_invalid_asset_collection_blocks_publication(tmp_path: Path, release_env: dict[str, str], filename: str, assets: object) -> None:
    change_release(tmp_path, "assets", assets, filename)
    result = run_guard(tmp_path, release_env, "publish")
    assert result.returncode != 0
    assert "release assets must be an array of named objects" in result.stderr
    assert_no_publication(tmp_path)
    if filename == "before.json":
        assert_no_remote_writes(tmp_path)
    else:
        assert "release upload" in (tmp_path / "calls").read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", ["before.json", "after.json"])
def test_missing_asset_collection_blocks_publication(tmp_path: Path, release_env: dict[str, str], filename: str) -> None:
    path = tmp_path / filename
    release = json.loads(path.read_text(encoding="utf-8"))
    del release["assets"]
    path.write_text(json.dumps(release), encoding="utf-8")
    assert run_guard(tmp_path, release_env, "publish").returncode != 0
    assert_no_publication(tmp_path)
    if filename == "before.json":
        assert_no_remote_writes(tmp_path)


@pytest.mark.parametrize("size", ["0", "26", None, True, {}, [], 0, -1, 1, 1.5])
def test_uploaded_asset_size_must_match_local_bytes(tmp_path: Path, release_env: dict[str, str], size: object) -> None:
    path = tmp_path / "after.json"
    release = json.loads(path.read_text(encoding="utf-8"))
    release["assets"][0]["size"] = size
    path.write_text(json.dumps(release), encoding="utf-8")
    result = run_guard(tmp_path, release_env, "publish")
    assert result.returncode != 0
    assert "different size or digest" in result.stderr
    assert "release upload" in (tmp_path / "calls").read_text(encoding="utf-8")
    assert_no_publication(tmp_path)


def test_duplicate_baseline_name_blocks_publication_even_if_one_copy_matches(tmp_path: Path, release_env: dict[str, str]) -> None:
    path = tmp_path / "after.json"
    release = json.loads(path.read_text(encoding="utf-8"))
    release["assets"].append({**release["assets"][0], "digest": "sha256:wrong"})
    path.write_text(json.dumps(release), encoding="utf-8")
    assert run_guard(tmp_path, release_env, "publish").returncode != 0
    assert_no_publication(tmp_path)


def test_unrelated_asset_names_are_data_not_filter_or_shell_code(tmp_path: Path, release_env: dict[str, str]) -> None:
    name = "other'\"; $(touch injected); [not a filter]"
    for filename in ("before.json", "after.json"):
        path = tmp_path / filename
        release = json.loads(path.read_text(encoding="utf-8"))
        release["assets"].append({"name": name})
        path.write_text(json.dumps(release), encoding="utf-8")
    result = run_guard(tmp_path, release_env, "publish")
    assert result.returncode == 0, result.stderr
    assert "--method PATCH" in (tmp_path / "calls").read_text(encoding="utf-8")
    assert not (tmp_path / "injected").exists()
