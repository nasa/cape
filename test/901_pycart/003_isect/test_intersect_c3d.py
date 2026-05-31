
# Standard library
import os

# Third-party imports
import testutils

# Local imports
from cape.pycart.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    "bullet.json",
    "bullet.xml",
    "bullet.tri",
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_02_run():
    # Instantiate
    cntl = Cntl()
    # Run first case
    cntl.SubmitJobs(I="0")
    # Get runner for that case
    runner = cntl.ReadCaseRunner(0)
    # Enter folder
    os.chdir(runner.root_dir)
    # Check for files
    assert os.path.isfile("Components.tri")
    assert os.path.isfile("Components.c.tri")
    assert os.path.isfile("Components.i.tri")
    # Check that Cart3D ran
    assert runner.get_iter() > 1

