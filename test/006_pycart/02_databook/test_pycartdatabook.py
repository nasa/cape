
# Standard Library
import os
import sys
import shutil

# Third-party
import testutils

# CAPE
import cape.pycart.cntl as pccntl


# Test Files
TEST_FILES = (
    "pyCart.json",
    "matrix.csv",
    os.path.join("poweroff", "m1.5a0.0b0.0", "bullet_no_base.dat"),
    os.path.join("poweroff", "m1.5a0.0b0.0", "case.json"),
    os.path.join("poweroff", "m1.5a0.0b0.0", "history.dat"),
    os.path.join("poweroff", "m1.5a0.0b0.0", "run.00.200"),
    "test.[0-9][0-9].out"
)

KW1 = {
    "I": [0],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW2 = {
    "I": [0],
    "fm": "bullet_no_base",
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    # Get cntl
    cntl = pccntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.01.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfm():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "aero_bullet_no_base.csv"))
    # Get cntl
    cntl = pccntl.Cntl()
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
