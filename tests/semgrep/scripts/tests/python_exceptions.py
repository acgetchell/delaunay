import subprocess
from unittest.mock import MagicMock, Mock


def catches_broad_exception() -> None:
    try:
        pass
    # ruleid: delaunay.python.no-broad-exception
    except Exception:
        pass


def catches_broad_exception_with_binding() -> None:
    try:
        pass
    # ruleid: delaunay.python.no-broad-exception
    except Exception as exc:
        raise RuntimeError("wrapped") from exc


def catches_specific_exception() -> None:
    try:
        pass
    # ok: delaunay.python.no-broad-exception
    except OSError:
        pass


def raises_raw_exception() -> None:
    # ruleid: delaunay.python.no-raw-exception-in-tests
    raise Exception("too broad")


def raises_specific_exception() -> None:
    # ok: delaunay.python.no-raw-exception-in-tests
    raise RuntimeError("specific failure")


def adhoc_mock_stdout() -> None:
    # ruleid: delaunay.python.no-adhoc-completedprocess-mock
    result = Mock()
    result.stdout = "ok"


def adhoc_mock_returncode() -> None:
    # ruleid: delaunay.python.no-adhoc-completedprocess-mock
    result = MagicMock()
    result.returncode = 0


def adhoc_mock_constructor_stdout() -> None:
    # ruleid: delaunay.python.no-adhoc-completedprocess-mock
    Mock(stdout="ok")


def adhoc_mock_constructor_returncode() -> None:
    # ruleid: delaunay.python.no-adhoc-completedprocess-mock
    MagicMock(returncode=0)


def typed_completed_process() -> subprocess.CompletedProcess[str]:
    # ok: delaunay.python.no-adhoc-completedprocess-mock
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")


def direct_subprocess_run() -> None:
    # ruleid: delaunay.python.no-direct-subprocess-run-outside-wrapper
    subprocess.run(["git", "status"], check=False)


# ruleid: delaunay.python.no-untyped-defs-in-scripts
def missing_return_annotation():
    return None


# ok: delaunay.python.no-untyped-defs-in-scripts
def explicit_return_annotation() -> None:
    return None
