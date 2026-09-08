
# Standard Library
import os
import sys

# Third-party
import pytest
import testutils

# CAPE
from cape.cfdx.options.actionopts import DEFAULT_ACTIONS
from cape.errors import CapeValueError
from cape.pycart.cli import main as pycart_main
from cape.pycart.cntl import Cntl


# Dir to case files
CASEDIR = os.path.join("poweroff", "m1.5a0.0b0.0")

# Test Files
TEST_FILES = (
    "pyCart.json",
    "c3dfunc.py",
    "Config.xml",
    "cap-patch.uh3d",
    "matrix.csv",
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    os.path.join(CASEDIR, "Components.i.triq"),
    "test.[0-9][0-9].out"
)


# Capture everything written to fds 1 and 2 (no capsys)
def _capture_output(func, *a, **kw):
    # Flush any pending Python-level writes
    sys.stdout.flush()
    sys.stderr.flush()
    # Redirect fds 1 and 2 to a pipe
    rfd, wfd = os.pipe()
    fd1, fd2 = os.dup(1), os.dup(2)
    try:
        os.dup2(wfd, 1)
        os.dup2(wfd, 2)
        try:
            # Run the function
            v = func(*a, **kw)
        finally:
            # Restore and close the write end
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(fd1, 1)
            os.dup2(fd2, 2)
            os.close(wfd)
        # Read everything written
        return v, os.read(rfd, 2**20).decode()
    finally:
        os.close(fd1)
        os.close(fd2)
        os.close(rfd)


# Script the user inputs
def _patch_input(monkeypatch, answers):
    anses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(anses))


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_default_approve():
    """Default 'approve' action marks cases PASS."""
    # Get cntl
    cntl = Cntl()
    # Perform default action
    inds = cntl.perform_action("approve", I=[0])
    # Check index list
    assert list(inds) == [0]
    # Check that the case was marked PASS
    assert cntl.x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_default_extend():
    """Default 'extend' and 'extend2' actions extend cases."""
    # Get cntl
    cntl = Cntl()
    # Get initial phase iters and nominal phase size
    n0 = cntl.read_case_json(0).get_PhaseIters(0)
    nj = cntl.get_phase_niter(0, 0)
    # Run default 'extend2' twice
    cntl.perform_action("extend2", I=[0])
    # Check that PhaseIters increased by two phase copies
    n1 = cntl.read_case_json(0).get_PhaseIters(0)
    assert n1 >= n0 + 2*nj


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_shell():
    """Shell actions replace '{I}' or append '-I' to the command."""
    # Get cntl
    cntl = Cntl()
    # Define a shell action w/ a placeholder
    cntl.opts["Actions"] = {
        "touch": "echo {I} > touched.txt",
        "append": "echo done > appended.txt",
    }
    # Run the shell action w/ placeholder
    cntl.perform_action("touch", I=[0])
    # Check output file got case indices
    with open("touched.txt") as f:
        assert f.read().strip() == "0"
    # Run the shell action w/o placeholder; '-I 0' gets appended
    cntl.perform_action("append", I=[0])
    # Check output file got command args appended
    with open("appended.txt") as f:
        assert f.read().strip().startswith("done")


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_cntl():
    """'cntl' actions call methods of the Cntl instance."""
    # Get cntl
    cntl = Cntl()
    # Define a custom action based on a Cntl method
    cntl.opts["Actions"] = {
        "markem": {"type": "cntl", "function": "MarkPASS"},
    }
    # Perform it
    cntl.perform_action("markem", I=[0])
    # Check that the case was marked PASS
    assert cntl.x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_cli():
    """'cli' actions call functions from cape.cfdx.cli."""
    # Get cntl
    cntl = Cntl()
    # Define a custom action based on a cli function
    cntl.opts["Actions"] = {
        "approve-cli": {"type": "cli", "function": "cape_approve"},
    }
    # Perform it
    cntl.perform_action("approve-cli", I=[0])
    # Re-read run matrix from file to check for PASS mark
    x = Cntl().x
    assert x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_shell_kwargs():
    """Shell actions *args* are not used; kwargs checked for cntl."""
    # Get cntl
    cntl = Cntl()
    # Define extension with explicit extend count
    cntl.opts["Actions"] = {
        "extend3": {
            "type": "cntl",
            "function": "ExtendCases",
            "kwargs": {"extend": 2},
        },
    }
    # Get initial phase iters and nominal phase size
    n0 = cntl.read_case_json(0).get_PhaseIters(0)
    nj = cntl.get_phase_niter(0, 0)
    # Perform custom extension
    cntl.perform_action("extend3", I=[0])
    # Check that PhaseIters increased by two phase copies
    n1 = cntl.read_case_json(0).get_PhaseIters(0)
    assert n1 >= n0 + 2*nj


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_simul():
    """Actions sharing an 'index' run together."""
    # Get cntl
    cntl = Cntl()
    # Define three actions; last two share an index
    cntl.opts["Actions"] = {
        "combo": [
            "echo first > first.txt",
            {"function": "echo {I} > simul-a.txt", "index": 1},
            {"function": "echo {I} > simul-b.txt", "index": 1},
        ],
    }
    # Perform actions; simultaneous ones must not interfere
    cntl.perform_action("combo", I=[0])
    # Check that all three actions ran
    assert os.path.isfile("first.txt")
    with open("simul-a.txt") as f:
        assert f.read().strip() == "0"
    with open("simul-b.txt") as f:
        assert f.read().strip() == "0"


def test_suppress_output():
    """_suppress_output silences writes to fds 1 and 2."""
    # Local part-level import
    from cape.cfdx.cntl import _suppress_output
    # Capture literal fd writes (bypasses pytest's sys-level capture)
    _, out = _capture_output(_noisy_func)
    # Check both streams
    assert "stdout visible" in out
    assert "stderr visible" in out
    # Capture again, now w/ suppression
    _, out = _capture_output(_noisy_func, out=_suppress_output)
    # Nothing should be captured
    assert out == ""


# Write to fds 1 and 2, optionally wrapped in a context manager
def _noisy_func(out=None):
    # Flush pytest's buffers to keep writes ordered
    sys.stdout.flush()
    sys.stderr.flush()
    # Get context manager
    ctx = out() if out is not None else _nullctx()
    # Write to both fds inside the context
    with ctx:
        os.write(1, b"stdout visible\n")
        os.write(2, b"stderr visible\n")


# Trivial context manager
class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_unknown():
    """Unknown action names raise CapeValueError."""
    # Get cntl
    cntl = Cntl()
    # Should raise immediately
    with pytest.raises(CapeValueError):
        cntl.perform_action("bogus", I=[0])


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_cli_badfunc():
    """'cli' actions must name actual cape.cfdx.cli functions."""
    # Get cntl
    cntl = Cntl()
    # Define an action w/ a bogus function name
    cntl.opts["Actions"] = {
        "bad": {"type": "cli", "function": "cape_bogus"},
    }
    # Should raise
    with pytest.raises(CapeValueError):
        cntl.perform_action("bad", I=[0])


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_nofunction():
    """Actions without a 'function' raise CapeValueError."""
    # Get cntl
    cntl = Cntl()
    # Define an action w/ no function/command
    cntl.opts["Actions"] = {
        "bad": {"type": "shell"},
    }
    # Should raise
    with pytest.raises(CapeValueError):
        cntl.perform_action("bad", I=[0])


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_replaces_default(monkeypatch):
    """Custom 'approve' replaces (not extends) the default action."""
    # Script the user input
    _patch_input(monkeypatch, ["a"])
    # Get cntl w/ custom approve action
    cntl = Cntl()
    # Read default action for later
    assert DEFAULT_ACTIONS["approve"][0]["function"] == "MarkPASS"
    # Redefine the default
    cntl.opts["Actions"] = {"approve": "echo custom {I} > custom.txt"}
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["approve"] == [0]
    # Custom command ran; default MarkPASS did *not*
    assert os.path.isfile("custom.txt")
    assert not cntl.x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_actions_usertool(monkeypatch):
    """UserTools in 'Actions' section show up in dispatch."""
    # Script the user input
    _patch_input(monkeypatch, ["t", "0"])
    # Get cntl
    cntl = Cntl()
    # Add a new-style user tool
    cntl.opts["Actions"] = {
        "UserTools": ["action-tool"],
        "action-tool": "echo {I} > action-tool-out.txt",
    }
    # Check tool registration
    tools = cntl.get_user_tools()
    assert tools == {"action-tool": ("action", "action-tool")}
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["tools"] == {"action-tool": [0]}
    # Check that the tool ran with the right case indices
    with open("action-tool-out.txt") as f:
        assert f.read().strip() == "0"


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_mixed_usertools():
    """New-style and legacy UserTools are merged."""
    # Get cntl
    cntl = Cntl()
    # Add tools in both places
    cntl.opts["UserTools"] = {"legacy-tool": "echo {I} > legacy.txt"}
    cntl.opts["Actions"] = {
        "UserTools": ["action-tool"],
        "action-tool": "echo {I} > action.txt",
    }
    # Check tool registration: new-style first, then legacy
    tools = cntl.get_user_tools()
    assert list(tools) == ["action-tool", "legacy-tool"]
    assert tools["action-tool"] == ("action", "action-tool")
    assert tools["legacy-tool"] == ("shell", "echo {I} > legacy.txt")


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_tool_actions(monkeypatch):
    """Actions-type tools run through the action system."""
    # Script the user input
    _patch_input(monkeypatch, ["t", "1"])
    # Get cntl
    cntl = Cntl()
    # Add one of each kind of tool
    cntl.opts["UserTools"] = {"legacy-tool": "echo {I} > legacy.txt"}
    cntl.opts["Actions"] = {
        "UserTools": ["action-tool"],
        "action-tool": "echo {I} > action.txt",
    }
    # Run the dispatch; select second tool (legacy)
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["tools"] == {"legacy-tool": [0]}
    # Check that the legacy tool ran
    with open("legacy.txt") as f:
        assert f.read().strip() == "0"


@testutils.run_sandbox(__file__, TEST_FILES)
def test_perform_cmdline():
    """The 'cape perform' command takes an action and case subset."""
    # Custom action in the JSON opts read by the CLI
    cntl = Cntl()
    cntl.opts["Actions"] = {
        "report-approve": "echo approved {I} >> report-approve.txt",
    }
    cntl.opts.write_jsonfile("pyCart.json")
    # Perform a custom action via the CLI
    ierr = pycart_main(["pycart", "perform", "report-approve", "-I", "0"])
    assert ierr == 0
    with open("report-approve.txt") as f:
        assert f.read().strip() == "approved 0"
    # Perform the default 'approve' action via the CLI
    ierr = pycart_main(["pycart", "perform", "approve", "-I", "0"])
    assert ierr == 0
    assert Cntl().x.PASS[0]
