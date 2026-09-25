# -*- coding: utf-8 -*-

# Standard library
import os
import shlex
import sys

# Third-party imports
import testutils


# List of help commands to try
CMD_LIST_HELP = (
    "cape.gruvoc",
    "cape.gruvoc --help",
    "cape.gruvoc help",
)

# List of sub-command help commands
CMD_LIST_SUBHELP = (
    "cape.gruvoc convert --help",
    "cape.gruvoc print --help",
    "cape.gruvoc small-vols --help",
)

# List of file globs to copy into sandbox
TEST_FILES = (
    "arrow.uh3d",
    "test.[0-9][0-9].out",
)


# Test 'gruvoc' help messages
@testutils.run_sandbox(__file__, TEST_FILES)
def test_help():
    # Loop through commands
    for j, cmdj in enumerate(CMD_LIST_HELP):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, _, ierr = testutils.call_o(cmdlistj)
        # Check return code
        assert ierr == 0
        # Check output
        result = testutils.compare_files(stdout, "test.01.out")
        assert result.line1 == result.line2


# Test 'gruvoc CMD --help'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_subhelp():
    # Loop through commands
    for j, cmdj in enumerate(CMD_LIST_SUBHELP):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, _, ierr = testutils.call_o(cmdlistj)
        # Check return code
        assert ierr == 0
        # Name of file with target output
        ftarg = "test.%02i.out" % (j + 2)
        # Check output
        result = testutils.compare_files(stdout, ftarg)
        assert result.line1 == result.line2


# Test 'gruvoc' with an unrecognized command
@testutils.run_sandbox(__file__, TEST_FILES)
def test_badcmd():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.gruvoc", "not-a-command"]
    # Run the command
    stdout, _, ierr = testutils.call_o(cmdlist)
    # Check return code
    assert ierr == 0
    # Check output
    result = testutils.compare_files(stdout, "test.05.out")
    assert result.line1 == result.line2


# Test 'gruvoc print' with and without ``-h``
@testutils.run_sandbox(__file__, TEST_FILES)
def test_print():
    # Loop through human-readable options
    for j, cmdj in enumerate((
            "cape.gruvoc print arrow.uh3d",
            "cape.gruvoc print -h arrow.uh3d")):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, _, ierr = testutils.call_o(cmdlistj)
        # Check return code
        assert ierr == 0
        # Name of file with target output
        ftarg = "test.%02i.out" % (j + 6)
        # Check output
        result = testutils.compare_files(stdout, ftarg)
        assert result.line1 == result.line2


# Test 'gruvoc small-vols'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_small_vols():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.gruvoc", "small-vols", "arrow.uh3d"]
    # Run the command
    stdout, _, ierr = testutils.call_o(cmdlist)
    # Check return code
    assert ierr == 0
    # Check output
    result = testutils.compare_files(stdout, "test.08.out")
    assert result.line1 == result.line2


# Test 'gruvoc convert'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_convert():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m"] + shlex.split(
        "cape.gruvoc convert -v arrow.uh3d arrow.lb8.ugrid")
    # Run the command
    _, _, ierr = testutils.call_o(cmdlist)
    # Check return code
    assert ierr == 0
    # Check that output file was created with nonzero size
    assert os.path.isfile("arrow.lb8.ugrid")
    assert os.stat("arrow.lb8.ugrid").st_size > 0


if __name__ == "__main__":
    test_help()
    test_subhelp()
    test_badcmd()
    test_print()
    test_small_vols()
    test_convert()
