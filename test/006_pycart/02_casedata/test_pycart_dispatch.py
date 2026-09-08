
# Standard Library
import os

# Third-party
import pytest
import testutils

# CAPE
from cape.errors import CapeValueError
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


# Add UserTools for dispatch tests
def _add_user_tools(cntl, toolcmd="echo {I} > tools-out.txt"):
    # Add a user tool
    if toolcmd is not None:
        cntl.opts["UserTools"] = {"echo-tool": toolcmd}


# Script the user inputs
def _patch_input(monkeypatch, answers):
    anses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(anses))


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_approve(monkeypatch):
    """Answer 'a' to approve case after seeing status."""
    # No display, just scripted answers
    _patch_input(monkeypatch, ["a"])
    # Get cntl
    cntl = Cntl()
    # Add a user tool
    _add_user_tools(cntl)
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["approve"] == [0]
    assert reviews["extend"] == []
    # Check that the case was marked PASS
    assert cntl.x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_extend(monkeypatch):
    """Answer 'e' to extend case by one phase copy."""
    # No display, just scripted answers
    _patch_input(monkeypatch, ["e"])
    # Get cntl
    cntl = Cntl()
    # Add a user tool
    _add_user_tools(cntl)
    # Get initial phase iters
    n0 = cntl.read_case_json(0).get_PhaseIters(0)
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["extend"] == [0]
    # Check that PhaseIters increased
    n1 = cntl.read_case_json(0).get_PhaseIters(0)
    assert n1 > n0


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_extend2(monkeypatch):
    """Answer 'e2' to extend case by two phase copies."""
    # No display, just scripted answers
    _patch_input(monkeypatch, ["e2"])
    # Get cntl
    cntl = Cntl()
    # Add a user tool
    _add_user_tools(cntl)
    # Get initial phase iters and nominal phase size
    n0 = cntl.read_case_json(0).get_PhaseIters(0)
    nj = cntl.get_phase_niter(0, 0)
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["extend"] == []
    assert reviews["extend2"] == [0]
    # Check that PhaseIters increased by two phase copies
    n1 = cntl.read_case_json(0).get_PhaseIters(0)
    assert n1 >= n0 + 2*nj


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_default_approve(monkeypatch):
    """Blank input accepts default (approve)."""
    # No display, just scripted answers
    _patch_input(monkeypatch, [""])
    # Get cntl
    cntl = Cntl()
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["approve"] == [0]
    assert cntl.x.PASS[0]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_prompt(monkeypatch):
    """Prompt offers approve but not next; bad input reprompts."""
    # Script the user input
    _patch_input(monkeypatch, ["n", "s"])
    # Get cntl
    cntl = Cntl()
    # Answer "n" not allowed w/ single decision (reprompt), then "s"
    action = cntl._prompt_dispatch(0, "case/name", "DONE", {})
    assert action == "skip"


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_usertool(monkeypatch):
    """Answer 't' then select tool; tool runs once with collected {I}."""
    # No display, just scripted answers
    _patch_input(monkeypatch, ["t", "0"])
    # Get cntl
    cntl = Cntl()
    # Add a user tool
    _add_user_tools(cntl)
    # Run the dispatch
    reviews = cntl.DispatchCases(I=[0])
    # Check recorded decision
    assert reviews["tools"] == {"echo-tool": [0]}
    # Check that the tool ran with the right case indices
    with open("tools-out.txt") as f:
        assert f.read().strip() == "0"


@testutils.run_sandbox(__file__, TEST_FILES)
def test_dispatch_usertool_missing_placeholder(monkeypatch):
    """UserTools commands must contain an '{I}' placeholder."""
    # No display, just scripted answers
    _patch_input(monkeypatch, ["t", "0"])
    # Get cntl
    cntl = Cntl()
    # Add a user tool w/ bad cmd
    _add_user_tools(cntl, toolcmd="echo no-placeholder")
    # Run the dispatch; should fail on placeholder check
    with pytest.raises(CapeValueError):
        cntl.DispatchCases(I=[0])
