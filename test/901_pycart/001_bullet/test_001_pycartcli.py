
# Standard library
import sys

# Third-party imports
import testutils

# Local imports
import cape.pycart.cntl


# List of commands to try
CMD_LIST = (
    "pycart -c",
    "cape -c --filter b2",
    "cape -c --cons 'beta==2,Mach%1==0.5'",
    "cape -c --glob 'poweroff/m0*'",
    "cape -c --re 'm.\\.5.*b2'",
    "cape -c -I 2:5,7,18:",
    "cape -c -I 15: --cons Mach%1=0.5 --re b2",
)

# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    "Config.xml",
    "bullet.tri",
    "test.[0-9][0-9].out"
)


# Test 'cape -c'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_c():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pycart", "-c"]
    # Run the command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Check outout
    result = testutils.compare_files(stdout, "test.01.out")
    assert result.line1 == result.line2


# Run a case
@testutils.run_sandbox(__file__, fresh=False)
def test_02_run():
    # Instantiate
    cntl = cape.pycart.cntl.Cntl()
    # Run first case
    cntl.SubmitJobs(I="0")
    # Collect aero
    cntl.cli(fm=True, I="0")

