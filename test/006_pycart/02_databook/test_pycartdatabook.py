
# Standard Library
import os
import shutil

# Third-party
import testutils

# CAPE
from cape.pycart.cntl import Cntl
from cape.dkit.rdb import DataKit


# Dir to case files
CASEDIR = os.path.join("poweroff", "m1.5a0.0b0.0")

# Test Files
TEST_FILES = (
    "pyCart.json",
    "c3dfunc.py",
    "matrix.csv",
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    "test.[0-9][0-9].out"
)


# Test Files
TEST_FILES2 = (
    os.path.join(CASEDIR, "Components.i.triq"),
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    "pyCart.json",
    "matrix.csv",
    "c3dfunc.py",
    "test.[0-9][0-9].out",
    os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
)

# Test Files
TEST_FILES3 = (
    "pyCart.json",
    "c3dfunc.py",
    "Config.xml",
    "cap-patch.uh3d",
    "matrix.csv",
    os.path.join(CASEDIR, "bullet_no_base.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "history.dat"),
    os.path.join(CASEDIR, "run.00.200"),
    os.path.join(CASEDIR, "Components.i.triq"),
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

KW3 = {
    "I": [0],
    "ll": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW4 = {
    "I": [0],
    "ll": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW5 = {
    "I": [0],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW6 = {
    "I": [0],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW7 = {
    "I": [0],
    "triqfm": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW8 = {
    "I": [0],
    "triqfm": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    # Get cntl
    cntl = Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    db = DataKit(dbout)
    # Compare output databook with reference result
    assert db["Mach"].size == 1
    assert db["CA"][0] > 0.7
    assert db["CA"][0] < 0.8


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfm():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "aero_bullet_no_base.csv"))
    # Get cntl
    cntl = Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW2)
    # Location of output databook
    dbout = os.path.join("data/aero_bullet_no_base.csv")
    # Read it
    db = DataKit(dbout)
    assert db["Mach"].size == 0


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_updatedatabookll():
    fll = os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
    # Ensure modifcation time of line load file is after triq file
    os.utime(fll, None)
    # Get cntl
    cntl = Cntl()
    # Call dbook updater
    cntl.UpdateLL(**KW3)
    # Location of output databook
    dbout = os.path.join("data/ll_bullet_total_LL.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.03.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_deletecasesll():
    fll = os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
    # Ensure modifcation time of line load file is after triq file
    os.utime(fll, None)
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy(
        "test.03.out",
        os.path.join("data", "ll_bullet_total_LL.csv"))
    # Get cntl
    cntl = Cntl()
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


@testutils.run_sandbox(__file__, TEST_FILES3)
def test_updatedatabookfunc():
    # Get cntl
    cntl = Cntl()
    # Call dbook updater
    cntl.UpdateDataBookPyFunc(**KW5)
    # Location of output databooks
    dbout1 = os.path.join("data/pyfunc_functest.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.05.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES3)
def test_deletecasesfunc():
    os.mkdir("data")
    # Use test.01.out as existing databook
    shutil.copy(
        "test.05.out",
        os.path.join("data", "pyfunc_functest.csv"))
    # Get cntl
    cntl = Cntl()
    # Call dbook updater
    cntl.UpdateDataBookPyFunc(**KW6)
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


@testutils.run_sandbox(__file__, TEST_FILES3)
def test_updatedatabooktriqfm():
    # Get cntl
    cntl = Cntl()
    # Call dbook updater
    cntl.UpdateTriqFM(**KW7)
    # Location of output databooks
    dbout1 = os.path.join("data/triqfm/triqfm_cap.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.07.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES3)
def test_deletecasestriqfm():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "triqfm"))
    # Use test.01.out as existing databook
    shutil.copy("test.07.out",
                os.path.join("data", "triqfm", "triqfm_cap.csv"))
    # Get cntl
    cntl = Cntl()
    # Call dbook updater
    cntl.UpdateTriqFM(**KW8)
    # Location of output databook
    dbout = os.path.join("data/triqfm/triqfm_cap.csv")
    # Location of old databook
    dbold = os.path.join("data/triqfm/triqfm_cap.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.08.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)

