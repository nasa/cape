
# Standard library
import os
import sys
import time

# Third-party imports
import testutils

# Local imports
from cape.pyfun.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "arrow.xml",
    "fun3d.nml",
    "arrow-*",
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_run():
    # Instantiate
    cntl = Cntl()
    # Case to run
    i = 0
    # Run first case
    cntl.SubmitJobs(I=str(i))
    # Name of case
    frun = cntl.x.GetFullFolderNames(i)
    # Path to preshell script and output file
    workerfile = os.path.join(frun, "cape-workershellcmd.log")
    logfile1 = os.path.join(frun, "cape", "cape-verbose.log")
    # Check that file exists
    assert os.path.isfile(workerfile)
    assert os.path.isfile(logfile1)
    # Check worker output something
    lines = open(workerfile).readlines()
    assert len(lines) >= 1
    # Check for log of Worker
    txt = open(logfile1).read()
    assert "starting 1 *WorkerShellCmds*" in txt

