
# Standard library
import sys

# Third-party imports
import testutils

# Local imports
from cape.pycart import cli
from cape.pycart.cntl import Cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    "aero.csh",
    "Config.xml",
    "bullet.tri",
    "test.[0-9][0-9].out"
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_02_run():
    # Instantiate
    cntl = Cntl()
    # Run first case
    cntl.SubmitJobs(I="0")
    # Collect aero
    cli.main(["pycart", "extract-fm", "-I", "0"])
    # Read databook
    cntl.ReadDataBook()
    # Get value
    CA = cntl.DataBook["bullet_no_base"]["CA"][0]
    # Test value
    assert abs(CA - 1.0) <= 0.2


if __name__ == "__main__":
    test_02_run()

