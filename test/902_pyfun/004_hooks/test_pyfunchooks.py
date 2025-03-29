
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
    "hookmod.py",
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
    logfile1 = os.path.join(frun, "cape-preshellcmd.log")
    logfile2 = os.path.join(frun, "cape-prepyfunc.log")
    logfile3 = os.path.join(frun, "cape", "cape-verbose.log")
    # Check that file exists
    assert os.path.isfile(shellfile)
    assert os.path.isfile(logfile1)
    assert os.path.isfile(logfile2)
    assert os.path.isfile(logfile3)
    # Check length
    lines = open(logfile1).readlines()
    assert len(lines) == 2
    # Check content from PythonFunc
    txt = open(logfile2).read()
    phases = [int(part) for part in txt.strip().split()]
    assert phases == [0, 1]
    # Make sure MYHOOK showed up in verbose log
    # (Tests "Type": "runner" for PrePythonFuncs)
    txt = open(logfile3).read()
    assert "MYHOOK" in txt



