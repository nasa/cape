
# Standard Library
import os

# Third-party
import pytest
import testutils

# CAPE
import cape.sysutils
from cape.errors import CapeNotSupportedError, CapeValueError
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


# Add a figure and UserTools to the "base" report
def _add_review_opts(cntl, toolcmd="echo {I} > tools-out.txt"):
    # Add one figure containing the "bullet_CA" subfigure
    cntl.opts["Report"]["base"]["Figures"] = ["fig_ca"]
    cntl.opts["Report"]["Figures"] = {"fig_ca": {"Subfigures": ["bullet_CA"]}}
    # Add a user tool
    if toolcmd is not None:
        cntl.opts["UserTools"] = {"echo-tool": toolcmd}


# Patch out terminal checks, image display, and get scripted inputs
def _patch_display(monkeypatch, answers):
    # Make terminals always support images
    monkeypatch.setattr(cape.sysutils, "terminal_image_supported", lambda: True)
    # Don't actually open any images
    monkeypatch.setattr(cape.sysutils, "open_img", lambda *a, **kw: "terminal")
    # Script the user input
    anses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(anses))


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_notsupported(monkeypatch):
    """ReviewCases raises if terminal does not support images."""
    # Make terminal unsupported
    monkeypatch.setattr(cape.sysutils, "terminal_image_supported", lambda: False)
    # Get cntl
    cntl = Cntl()
    # Should raise immediately
    with pytest.raises(CapeNotSupportedError):
        cntl.ReviewCases(I=[0])


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_approve(monkeypatch):
    """Answer 'a' to approve case at last subfigure."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["a"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Run the review; also check a tool w/ missing placeholder
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["approve"] == [0]
        assert reviews["extend"] == []
        # Check that the case was marked PASS
        assert cntl.x.PASS[0]
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_extend(monkeypatch):
    """Answer 'e' to extend case by one phase copy."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["e"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Get initial phase iters
        n0 = cntl.read_case_json(0).get_PhaseIters(0)
        # Run the review
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["extend"] == [0]
        # Check that PhaseIters increased
        n1 = cntl.read_case_json(0).get_PhaseIters(0)
        assert n1 > n0
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_extend2(monkeypatch):
    """Answer 'e2' to extend case by two phase copies."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["e2"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Get initial phase iters and nominal phase size
        n0 = cntl.read_case_json(0).get_PhaseIters(0)
        nj = cntl.get_phase_niter(0, 0)
        # Run the review
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["extend"] == []
        assert reviews["extend2"] == [0]
        # Check that PhaseIters increased by two phase copies
        n1 = cntl.read_case_json(0).get_PhaseIters(0)
        assert n1 >= n0 + 2*nj
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_atselect(monkeypatch):
    """Answer '@1' to select first option (approve at last subfig)."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["@1"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Run the review
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["approve"] == [0]
        assert cntl.x.PASS[0]
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_next_approve(monkeypatch):
    """Answer 'n' to advance subfigure, then 'a' to approve."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Script the user input
        anses = iter(["n", "", "a"])
        monkeypatch.setattr("builtins.input", lambda *a: next(anses))
        # Answer "n" when not on last subfigure
        action = cntl._prompt_review(0, "case/name", "sfig", False, {})
        assert action == "next"
        # Blank input accepts default ("next")
        action = cntl._prompt_review(0, "case/name", "sfig", False, {})
        assert action == "next"
        # Answer "a" to approve on last subfigure
        action = cntl._prompt_review(0, "case/name", "sfig", True, {})
        assert action == "approve"
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_tool_cancel(monkeypatch):
    """Blank input in tool menu cancels back to main prompt."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["t", "", "s"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Run the review; first 't' opens menu, blank cancels, 's' skips
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["skip"] == [0]
        assert reviews["tools"] == {}
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_usertool(monkeypatch):
    """Answer 't' then select tool; tool runs once with collected {I}."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["t", "0"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions
        _add_review_opts(cntl)
        # Run the review
        reviews = cntl.ReviewCases(I=[0])
        # Check recorded decision
        assert reviews["tools"] == {"echo-tool": [0]}
        # Check that the tool ran with the right case indices
        with open("tools-out.txt") as f:
            assert f.read().strip() == "0"
    finally:
        del os.environ["CAPE_CACHE_DIR"]


@testutils.run_sandbox(__file__, TEST_FILES)
def test_review_usertool_missing_placeholder(monkeypatch):
    """UserTools commands must contain an "{I}" placeholder."""
    # Set up cache directory in work folder
    os.environ["CAPE_CACHE_DIR"] = "cache"
    os.mkdir("cache")
    try:
        # No display, just scripted answers
        _patch_display(monkeypatch, ["t", "0"])
        # Get cntl
        cntl = Cntl()
        # Add figure and tool definitions, w/ bad tool cmd
        _add_review_opts(cntl, toolcmd="echo no-placeholder")
        # Run the review; should fail on placeholder check
        with pytest.raises(CapeValueError):
            cntl.ReviewCases(I=[0])
    finally:
        del os.environ["CAPE_CACHE_DIR"]
