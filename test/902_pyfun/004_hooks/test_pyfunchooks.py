
# Standard library
import os
import sys

# Third-party imports
import testutils

# Local imports
from cape.pyfun.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "bullet.xml",
    "fun3d.nml",
    "matrix.csv",
    "bullet-*",
    "preshell.sh",
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_run():
    # Instantiate
    cntl = Cntl()
    # Case to run
    i = 8
    # Run first case
    cntl.SubmitJobs(I=str(i))
    # Name of case
    frun = cntl.x.GetFullFolderNames(i)
    # Path to preshell script and output file
    shellfile = os.path.join(frun, "preshell.sh")
    logfile = os.path.join(frun, "cape-preshellcmd.log")
    # Check that file exists
    assert os.path.isfile(shellfile)
    assert os.path.isfile(logfile)
    # Check length
    lines = open(logfile).readlines()
    assert len(lines) == 2


