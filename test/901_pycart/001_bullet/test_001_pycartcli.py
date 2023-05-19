
# Standard library
import sys

# Third-party imports
import testutils

# Local imports
import cape.pycart.cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    "Config.xml",
    "bullet.tri",
    "test.[0-9][0-9].out"
)


# Test 'cape -c'
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_c():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pycart", "-c"]
    # Run the command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Check outout
    result = testutils.compare_files(stdout, "test.01.out")
    assert result.line1 == result.line2


# Run a case
@testutils.run_sandbox(__file__, fresh=False)
def test_02_run():
    # Instantiate
    cntl = cape.pycart.cntl.Cntl()
    # Run first case
    cntl.SubmitJobs(I="0")
    # Collect aero
    cntl.cli(fm=True, I="0")
    # Read databook
    cntl.ReadDataBook()
    # Get value
    CA = cntl.DataBook["bullet_no_base"]["CA"][0]
    # Test value
    assert abs(CA - 0.745) <= 0.001


if __name__ == "__main__":
    test_01_c()
    test_02_run()

