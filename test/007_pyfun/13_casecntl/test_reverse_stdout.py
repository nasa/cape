# Third-party
import pytest

# Local imports
from cape.pyfun.casecntl import CaseRunner


@pytest.mark.parametrize("contents", [b"", b"x", b"\n", b"\nStarting FUN3D\n"])
def test_startup_stdout_has_no_completed_iterations(tmp_path, contents):
    path = tmp_path / "fun3d.out"
    path.write_bytes(contents)
    runner = CaseRunner(str(tmp_path))
    assert runner.get_iter_active() == 0
    assert runner.get_iter_restart_stdout(str(path)) == 0


@pytest.mark.parametrize("ending", [b"", b"\n"])
def test_reverse_stdout_still_finds_latest_restart(tmp_path, ending):
    path = tmp_path / "fun3d.out"
    path.write_bytes(
        b"\ninserting current history iterations 10\n"
        b"inserting current history iterations 25\nfinishing" + ending)
    runner = CaseRunner(str(tmp_path))
    assert runner.get_iter_active() == 25
    assert runner.get_iter_restart_stdout(str(path)) == 25
