
# Third-party
import testutils

# Local imports
from cape.cfdx import cli
from cape.cfdx.cntl import Cntl


# Files to copy to sandbox
TEST_FILES = (
    "cape.json",
    "matrix.csv",
)


# Test pass/fail commands
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_pass():
    # Instantiate
    cntl = Cntl()
    # Pass case 5
    assert not cntl.x.PASS[5]
    cli.main(["cfdx", "-I", "5", "--PASS"])
    cntl = Cntl()
    assert cntl.x.PASS[5]
    # Do an error
    cli.main(["cfdx", "mark-error", "-I", "6"])
    cntl = Cntl()
    assert cntl.x.ERROR[6]
    # Unmark a case
    cli.main(["cfdx", "--unmark", "-I", "6"])
    cntl = Cntl()
    assert not cntl.x.ERROR[6]


if __name__ == "__main__":
    test_01_pass()

