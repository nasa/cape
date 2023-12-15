
# Third-party
import testutils

# Local imports
import cape.cntl


# Files to copy
TEST_FILES = (
    "cape.json",
    "BatchShell.json",
    "bullet.tri",
    "matrix.csv",
    "Config.xml"
)
TEST_DIRS = (
    "tools",
)


# Basic tests
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_01_cntl():
    # Instatiate
    cntl = cape.cntl.Cntl()
    # Test __repr__
    assert str(cntl) == "<cape.cntl.Cntl(nCase=20)>"
    # Test hook import
    assert "dac" in cntl.modules

