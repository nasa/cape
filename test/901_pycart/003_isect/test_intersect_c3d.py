
# Standard library

# Third-party imports
import testutils

# Local imports
from cape.pycart.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    "bullet.json",
    "bullet.tri",
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_02_run():
    # Instantiate
    cntl = Cntl()
    # Run first case
    cntl.SubmitJobs(I="0")
