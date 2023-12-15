
# Standard library
import os
import shlex
import sys

# Third-party imports
import testutils


# List of commands to try
CMD_LIST = (
    "cape -c",
    "cape -c -f cape-json.json",
    "cape -c -f cape-json.json --re a0",
    "cape -c -f cape-mixed.json",
    "cape -c -f cape-mixed.json --re a2",
)
CMD_RETURN = (True, False, False, False, False)

# List of file globs to copy into sandbox
TEST_FILES = (
    "cape-json.json",
    "cape-mixed.json",
    "matrix.csv",
    "test.[0-9][0-9].???"
)


# Test 'cape -c'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_c():
    # Loop through commands
    for j, cmdj in enumerate(CMD_LIST):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, stderr, ierr = testutils.call_oe(cmdlistj)
        # Name of file with target output
        if ierr:
            # Something went wrong
            ftarg = "test.%02i.err" % (j + 1)
            # Check STDERR target exists
            assert os.path.isfile(ftarg)
            # Compare STDERR
            result = testutils.compare_files(stderr, ftarg)
        else:
            # Ran succesfully
            ftarg = "test.%02i.out" % (j + 1)
            # Check STDOUT target exists
            assert os.path.isfile(ftarg)
            # Check outout
            result = testutils.compare_files(stdout, ftarg)
        assert result.line1 is result.line2 is None
