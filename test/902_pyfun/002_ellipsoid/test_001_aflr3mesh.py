
# Standard library
import os
import sys

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
    os.path.join("report", "NASA_logo.pdf"),
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_run():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Run a case
    cntl.SubmitJobs(I="6")


# Test existence of aflr out file
@testutils.run_sandbox(__file__, fresh=False)
def test_02_aflr():
    # Instantiate
    cntl = cape.pyfun.cntl.Cntl()
    # Get case folder
    case_folder = cntl.x.GetFullFolderNames(6)
    # Location of aflr3.out file
    aflr3_file = os.path.join(case_folder, "aflr3.out")
    # Check if file exists
    assert os.path.isfile(aflr3_file)


if __name__ == "__main__":
    test_01_run()
    test_02_aflr()

