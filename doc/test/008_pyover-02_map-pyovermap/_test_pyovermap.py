
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
    # Get list of mesh files
    assert cntl.GetMeshFileNames(0) == ["test.mixsur"]
    assert cntl.GetMeshFileNames(1) == ["test.usurp"]
    # Prepare files for both cases
    cntl.PrepareCase(0)
    cntl.PrepareCase(1)
    # Check that both have gone from status ``None`` to ``0``
    assert cntl.CheckCase(0) == 0
    assert cntl.CheckCase(1) == 0


if __name__ == "__main__":
    test_meshmap01()
