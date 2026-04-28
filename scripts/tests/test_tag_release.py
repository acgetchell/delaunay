"""Tests for tag_release.py — repo URL normalization and credential safety."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from tag_release import _get_repo_url, validate_semver

# ---------------------------------------------------------------------------
# _get_repo_url
# ---------------------------------------------------------------------------


def _fake_remote(url: str) -> subprocess.CompletedProcess[str]:
    """Return a mock CompletedProcess whose stdout is *url*."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=url + "\n")


class TestGetRepoUrl:
    """Tests for _get_repo_url normalization and credential refusal."""

    @patch("tag_release.run_git_command")
    def test_github_ssh(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("git@github.com:owner/repo.git")
        assert _get_repo_url() == "https://github.com/owner/repo"

    @patch("tag_release.run_git_command")
    def test_github_https(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("https://github.com/owner/repo.git")
        assert _get_repo_url() == "https://github.com/owner/repo"

    @patch("tag_release.run_git_command")
    def test_github_ssh_protocol(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("ssh://git@github.com/owner/repo.git")
        assert _get_repo_url() == "https://github.com/owner/repo"

    @patch("tag_release.run_git_command")
    def test_plain_non_github_url_returned(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("https://gitlab.com/owner/repo.git")
        assert _get_repo_url() == "https://gitlab.com/owner/repo.git"

    @patch("tag_release.run_git_command")
    def test_rejects_https_user_pass(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("https://user:token@github.com/owner/repo.git")
        with pytest.raises(ValueError, match="credentials"):
            _get_repo_url()

    @patch("tag_release.run_git_command")
    def test_rejects_https_user_only(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("https://user@github.com/owner/repo.git")
        with pytest.raises(ValueError, match="credentials"):
            _get_repo_url()

    @patch("tag_release.run_git_command")
    def test_rejects_ssh_style_non_github(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("deploy@gitlab.com:owner/repo.git")
        with pytest.raises(ValueError, match="credentials"):
            _get_repo_url()


# ---------------------------------------------------------------------------
# validate_semver
# ---------------------------------------------------------------------------


class TestValidateSemver:
    @pytest.mark.parametrize(
        "tag",
        ["v1.2.3", "v0.0.0", "v1.2.3-rc.1", "v1.2.3+build.42"],
    )
    def test_valid_tags(self, tag: str) -> None:
        validate_semver(tag)  # should not raise

    @pytest.mark.parametrize(
        "tag",
        ["1.2.3", "v1.2", "vx.y.z", "v01.2.3"],
    )
    def test_invalid_tags(self, tag: str) -> None:
        with pytest.raises(ValueError, match="SemVer"):
            validate_semver(tag)
