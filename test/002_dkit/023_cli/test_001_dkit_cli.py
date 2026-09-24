# -*- coding: utf-8 -*-

# Standard library
import shlex
import sys

# Third-party imports
import testutils


# List of help commands to try
CMD_LIST_HELP = (
    "cape.dkit",
    "cape.dkit -h",
    "cape.dkit help",
)

# List of sub-command help commands
CMD_LIST_SUBHELP = (
    "cape.dkit writedb -h",
    "cape.dkit vendorize -h",
    "cape.dkit quickstart -h",
)

# List of file globs to copy into sandbox
TEST_FILES = (
    "test.[0-9][0-9].out",
)


# Test 'dkit' help messages
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


# Test 'dkit CMD -h'
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


# Test 'dkit' with an unrecognized command
@testutils.run_sandbox(__file__, TEST_FILES)
def test_badcmd():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.dkit", "not-a-command"]
    # Run the command
    stdout, _, ierr = testutils.call_o(cmdlist)
    # Check return code
    assert ierr == 0
    # Check output
    result = testutils.compare_files(stdout, "test.05.out")
    assert result.line1 == result.line2


# Test 'dkit vendorize' in an empty folder
@testutils.run_sandbox(__file__, TEST_FILES)
def test_vendorize_nothing():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.dkit", "vendorize"]
    # Run the command
    stdout, _, ierr = testutils.call_o(cmdlist)
    # Check return code
    assert ierr == 0
    # Check output
    result = testutils.compare_files(stdout, "test.06.out")
    assert result.line1 == result.line2


if __name__ == "__main__":
    test_help()
    test_subhelp()
    test_badcmd()
    test_vendorize_nothing()
