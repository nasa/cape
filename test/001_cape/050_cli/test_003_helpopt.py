
# Standard library
import shlex
import sys

# Third-party imports
import testutils


# List of commands to try
CMD_LIST = (
    "cape help-opt .RunControl.PhaseIters",
    "cape help-opt -f cape.json .RunControl.PhaseIters",
    "cape help-opt -f cape.json --jq .Config.RefArea",
    "cape help-opt -f cape.json .RunControl --maxdepth 0",
)

# List of file globs to copy into sandbox
TEST_FILES = (
    "cape.json",
    "matrix.csv",
    "test.[0-9][0-9].out"
)


# Test 'cape help-opt'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_helpopt():
    # Loop through commands
    for j, cmdj in enumerate(CMD_LIST):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, _, _ = testutils.call_o(cmdlistj)
        # Name of file with target output
        ftarg = "test.%02i.out" % (j + 8)
        # Check outout
        result = testutils.compare_files(stdout, ftarg)
        assert result.line1 == result.line2


if __name__ == "__main__":
    test_helpopt()
