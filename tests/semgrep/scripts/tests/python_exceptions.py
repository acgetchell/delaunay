import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock

UTF8 = "utf-8"


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


def implicit_path_read_text_encoding(path: Path) -> None:
    # ruleid: delaunay.python.explicit-path-text-encoding-in-tests
    path.read_text()


def implicit_path_write_text_encoding(path: Path) -> None:
    # ruleid: delaunay.python.explicit-path-text-encoding-in-tests
    path.write_text("Time: [1.0, 1.0, 1.0] µs\n")


def explicit_path_text_encoding(path: Path) -> None:
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.read_text(encoding="utf-8")
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.write_text("Time: [1.0, 1.0, 1.0] µs\n", encoding="utf-8")
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.read_text(encoding='utf-8')
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.write_text("Time: [1.0, 1.0, 1.0] µs\n", encoding='utf-8')
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.read_text(encoding=UTF8)
    # ok: delaunay.python.explicit-path-text-encoding-in-tests
    path.write_text("Time: [1.0, 1.0, 1.0] µs\n", encoding=UTF8)


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
