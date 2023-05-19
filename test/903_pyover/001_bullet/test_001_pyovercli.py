
# Standard library
import sys

# Third-party imports
import testutils

# Local imports
import cape.pyover.cntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyOver.json",
    "test.[0-9][0-9].out"
)
TEST_DIRS = (
    "common",
    "inputs"
)


# Run a case
@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_01_run():
    # Instantiate
    cntl = cape.pyover.cntl.Cntl()
    # Run first case
    cntl.SubmitJobs(I="1")


# Test 'pyover -c'
@testutils.run_sandbox(__file__, fresh=False)
def test_02_c():
    # Split command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyover", "-c", "-I", "1"]
    # Run the command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Check outout
    result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    assert result.line1 == result.line2


# Collect aero
@testutils.run_sandbox(__file__, fresh=False)
def test_03_fm():
    # Instantiate
    cntl = cape.pyover.cntl.Cntl()
    # Collect aero
    cntl.cli(fm=True, I="1")
    # Read databook
    cntl.ReadDataBook()
    # Get value
    CN = cntl.DataBook["bullet_no_base"]["CN"][0]
    # Test value
    assert abs(CN - 0.21) <= 0.02


if __name__ == "__main__":
    test_01_run()

