
# Standard library
import sys
import os
import shutil

# Third-party imports
import testutils


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyOver.json",
    "matrix.csv",
    "test.[0-9][0-9].out"
)
TEST_DIRS = (
    # "inputs",
    "poweroff"
)


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_updatedatabookfm():
    # Split update command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyover", "-I", "1", "--fm",
               "bullet_no_base"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.01.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_deletecasesfm():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "aero_bullet_no_base.csv"))
    # Split delete command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyover", "-I", "1", "--fm",
               "bullet_no_base", "--delete"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Location of old databook
    dbold = os.path.join("data/aero_bullet_no_base.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.02.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)
