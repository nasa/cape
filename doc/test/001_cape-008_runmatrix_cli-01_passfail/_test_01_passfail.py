
# Third-party
import testutils

# Local imports
import cape.cntl


# Files to copy to sandbox
TEST_FILES = (
    "cape.json",
    "matrix.csv",
)


# Test pass/fail commands
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_pass():
    # Instantiate
    cntl = cape.cntl.Cntl()
    # Pass case 5
    assert not cntl.x.PASS[5]
    cntl.cli(I=5, PASS=True)
    assert cntl.x.PASS[5]
    # Do an error
    cntl.cli(I=6, ERROR=True)
    assert cntl.x.ERROR[6]
    # Unmark a case
    cntl.cli(I=6, unmark=True)
    assert not cntl.x.ERROR[6]
