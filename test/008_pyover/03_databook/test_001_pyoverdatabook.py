
# Standard library
import sys
import os
import shutil

# Third-party imports
import testutils

# CAPE
import cape.pyover.cntl as pocntl


# List of file globs to copy into sandbox
TEST_FILES = (
    "pyOver.json",
    "matrix.csv",
    "test.[0-9][0-9].out"
)
TEST_DIRS = (
    "poweroff"
)


KW1 = {
    "I": [1],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW2 = {
    "I": [1],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_updatedatabookfm():
    # Get cntl
    cntl = pocntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
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
    # Get cntl
    cntl = pocntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW2)
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
