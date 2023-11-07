
# Third-party imports
import testutils

# Local imports
from cape.pyover.cntl import Cntl


# List of files to cpy
COPY_FILES = (
    "matrix.csv",
    "overflow.inp",
    "pyOver.json",
)
COPY_DIRS = (
    "common_inflow",
)


# Initialize
@testutils.run_sandbox(__file__, COPY_FILES, COPY_DIRS)
def test_meshmap01():
    # Instantiate control class
    cntl = Cntl()


if __name__ == "__main__":
    test_meshmap01()
