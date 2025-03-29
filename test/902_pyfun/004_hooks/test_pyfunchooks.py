
# Standard library
import os
import sys

# Third-party imports
import testutils

# Local imports
from cape.pyfun.cntl import Cntl
from cape.pyfun import cli


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "bullet.xml",
    "fun3d.nml",
    "matrix.csv",
    "bullet-*",
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_run():
    # Instantiate
    cntl = Cntl()
    # Run first case
    cntl.SubmitJobs(I="8")


