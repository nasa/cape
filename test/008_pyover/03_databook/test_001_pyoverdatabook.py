
# Standard library
import os
import shutil

# Third-party imports
import testutils

# CAPE
import cape.pyover.cntl as pocntl


# Dir to case files
CASEDIR = os.path.join("poweroff", "m0.8a4.0b0.0")

# List of file globs to copy into sandbox
TEST_FILES = (
    "pyOver.json",
    "matrix.csv",
    "oflfunc",
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

KW3 = {
    "I": [1],
    "ll": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW4 = {
    "I": [1],
    "ll": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW5 = {
    "I": [1],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW6 = {
    "I": [1],
    "dbpyfunc": True,
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


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_updatedatabookll():
    fll = os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
    # Ensure modifcation time of line load file is after triq file
    os.utime(fll, None)
    # Get cntl
    cntl = pocntl.Cntl()
    # Call dbook updater
    cntl.UpdateLL(**KW3)
    # Location of output databook
    dbout = os.path.join("data/ll_bullet_total_LL.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.03.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_deletecasesll():
    fll = os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
    # Ensure modifcation time of line load file is after triq file
    os.utime(fll, None)
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.03.out", os.path.join("data",
                                            "ll_bullet_total_LL.csv"))
    # Get cntl
    cntl = pocntl.Cntl()
    # Call dbook updater
    cntl.UpdateLL(**KW4)
    # Location of output databook
    dbout = os.path.join("data/ll_bullet_total_LL.csv")
    # Location of old databook
    dbold = os.path.join("data/ll_bullet_total_LL.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.04.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_updatedatabookfunc():
    # Get cntl
    cntl = pocntl.Cntl()
    # Call dbook updater
    cntl.UpdateDBPyFunc(**KW5)
    # Location of output databooks
    dbout1 = os.path.join("data/pyfunc_functest.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.05.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
def test_deletecasesfunc():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.05.out",
                os.path.join("data", "pyfunc_functest.csv"))
    # Get cntl
    cntl = pocntl.Cntl()
    # Call dbook updater
    cntl.UpdateDBPyFunc(**KW6)
    # Location of output databook
    dbout = os.path.join("data/pyfunc_functest.csv")
    # Location of old databook
    dbold = os.path.join("data/pyfunc_functest.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.06.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)

