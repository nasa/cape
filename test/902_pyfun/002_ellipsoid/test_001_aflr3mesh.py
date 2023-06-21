
# Standard library
import os

# Third-party imports
import testutils

# Local imports
import cape.pyfun.cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyFun.json",
    "fun3d.nml",
    "matrix.csv",
    "Ellipsoid-*",
)


# File names
LOG_FILE = "aflr3.out"
MESH_PREFIX = "ellipsoid"


# Test AFLR3 execution
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_aflr3():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Run a case
    cntl.SubmitJobs(I="6")
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(6)
    # Full path to ``aflr3.out`` and others
    log_file = os.path.join(case_folder, LOG_FILE)
    mesh_file = os.path.join(case_folder, f"{MESH_PREFIX}.ugrid")
    # Check if files exist
    assert os.path.isfile(log_file)
    assert os.path.isfile(mesh_file)


if __name__ == "__main__":
    test_01_aflr3()

