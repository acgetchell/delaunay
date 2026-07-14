"""Tests for tag_release.py — repo URL normalization and credential safety."""

import subprocess
from pathlib import PureWindowsPath
from unittest.mock import patch

import pytest

from tag_release import _get_repo_url, create_tag, main, validate_semver

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
    def test_plain_non_github_url_rejected(self, mock_git) -> None:
        mock_git.return_value = _fake_remote("https://gitlab.com/owner/repo.git")
        with pytest.raises(ValueError, match="Origin remote"):
            _get_repo_url()

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
        with pytest.raises(ValueError, match="Origin remote"):
            _get_repo_url()

    @pytest.mark.parametrize(
        "raw",
        [
            "https://github.com/owner/repo.git?token=secret-token",
            "https://github.com/owner/repo.git#secret-token",
        ],
    )
    @patch("tag_release.run_git_command")
    def test_rejects_query_and_fragment_without_echoing_them(self, mock_git, raw: str) -> None:
        mock_git.return_value = _fake_remote(raw)

        with pytest.raises(ValueError, match="query parameters or fragments") as exc_info:
            _get_repo_url()

        assert "secret-token" not in str(exc_info.value)
        assert raw not in str(exc_info.value)

    @pytest.mark.parametrize(
        "raw",
        [
            "http://github.com/owner/repo.git",
            "https://example.com/owner/repo.git",
            "git@example.com:owner/repo.git",
            "ssh://developer@github.com/owner/repo.git",
            "https://github.com/owner/repo/extra.git",
            "file:///tmp/repo.git",
            "/home/example/repo.git",
            "../repo.git",
            "https://[malformed",
        ],
    )
    @patch("tag_release.run_git_command")
    def test_rejects_unsupported_remotes_without_echoing_them(self, mock_git, raw: str) -> None:
        mock_git.return_value = _fake_remote(raw)

        with pytest.raises(ValueError, match="Origin remote") as exc_info:
            _get_repo_url()

        assert raw not in str(exc_info.value)


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


# ---------------------------------------------------------------------------
# create_tag
# ---------------------------------------------------------------------------


class TestCreateTag:
    def test_truncated_message_uses_posix_source_url(
        self,
        tmp_path,
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        with (
            patch("tag_release._tag_exists", return_value=False),
            patch("tag_release.find_changelog", return_value=changelog),
            patch("tag_release.extract_changelog_section") as mock_extract_changelog,
            patch("tag_release._github_anchor", return_value="v123"),
            patch("tag_release._get_repo_url", return_value="https://github.com/owner/repo"),
            patch("tag_release.run_git_command_with_input") as mock_run_git_with_input,
        ):
            mock_extract_changelog.return_value = (
                "x" * 125_001,
                PureWindowsPath("docs\\archive\\changelog\\1.2.md"),
            )

            create_tag("v1.2.3")

        tag_message = mock_run_git_with_input.call_args.kwargs["input_data"]
        assert "<https://github.com/owner/repo/blob/v1.2.3/docs/archive/changelog/1.2.md#v123>" in tag_message
        assert "docs\\archive\\changelog\\1.2.md" not in tag_message

    def test_force_replaces_existing_tag_without_delete(self, tmp_path) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        with (
            patch("tag_release._tag_exists", return_value=True),
            patch("tag_release.find_changelog", return_value=changelog),
            patch("tag_release.extract_changelog_section", return_value=("## v1.2.3\n\n- Fixed\n", changelog)),
            patch("tag_release.run_git_command_with_input") as mock_run_git_with_input,
        ):
            create_tag("v1.2.3", force=True)

        mock_run_git_with_input.assert_called_once()
        assert mock_run_git_with_input.call_args.args[0] == ["tag", "-f", "-a", "v1.2.3", "-F", "-", "--cleanup=verbatim"]

    def test_invalid_remote_does_not_replace_existing_tag(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        changelog = tmp_path / "CHANGELOG.md"
        marker = "release-token"
        with (
            patch("tag_release._tag_exists", return_value=True),
            patch("tag_release.find_changelog", return_value=changelog),
            patch("tag_release.extract_changelog_section", return_value=("x" * 125_001, changelog)),
            patch("tag_release.run_git_command") as mock_run_git,
            patch("tag_release.run_git_command_with_input") as mock_run_git_with_input,
        ):
            mock_run_git.return_value.stdout = f"https://github.com/owner/repo.git?token={marker}"

            with pytest.raises(ValueError, match="query parameters") as exc_info:
                create_tag("v1.2.3", force=True)

        mock_run_git_with_input.assert_not_called()
        output = capsys.readouterr()
        assert marker not in str(exc_info.value)
        assert marker not in output.out
        assert marker not in output.err


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def test_main_handles_git_timeout(capsys) -> None:
    with (
        patch("sys.argv", ["tag-release", "v1.2.3"]),
        patch(
            "tag_release.create_tag",
            side_effect=subprocess.TimeoutExpired(cmd=["git", "tag"], timeout=30),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()

    assert exc_info.value.code == 1
    assert "Error: Command '['git', 'tag']' timed out after 30 seconds" in capsys.readouterr().err
