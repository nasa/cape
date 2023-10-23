
# Standard library
import os

# Third-party
import testutils

# Local imports
import cape.cntl


# Files to copy
TEST_FILES = (
    "cape.json",
    "matrix.csv",
    "Config.xml"
)


# Basic tests
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_cntl():
    # Instatiate
    cntl = cape.cntl.Cntl()
    # Pick a case
    i = 14
    # Get name of case
    frun = cntl.x.GetFullFolderNames(i)
    # Make sure it's got a lot of / in it
    assert frun.count('/') == 3
    # Create a folder
    cntl.make_case_folder(i)
    # Make sure expected folder exists
    # (Tests making arbitrary-depth case folders)
    assert os.path.isdir(frun)

