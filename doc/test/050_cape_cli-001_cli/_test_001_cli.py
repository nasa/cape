
# Standard library
import shlex
import sys

# Third-party imports
import testutils


# List of commands to try
CMD_LIST = (
    "cape -c",
    "cape -c --filter b2",
    "cape -c --cons 'beta==2,Mach%1==0.5'",
    "cape -c --glob 'poweroff/m0*'",
    "cape -c --re 'm.\\.5.*b2'",
    "cape -c -I 2:5,7,18:",
    "cape -c -I 15: --cons Mach%1=0.5 --re b2",
)

# List of file globs to copy into sandbox
TEST_FILES = (
    "cape.json",
    "matrix.csv",
    "test.[0-9][0-9].out"
)


# Test 'cape -c'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_c():
    # Loop through commands
    for j, cmdj in enumerate(CMD_LIST):
        # Split command and add `-m` prefix
        cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
        # Run the command
        stdout, _, _ = testutils.call_o(cmdlistj)
        # Name of file with target output
        ftarg = "test.%02i.out" % (j + 1)
        # Check outout
        result = testutils.compare_files(stdout, ftarg)
        assert result.line1 == result.line2
