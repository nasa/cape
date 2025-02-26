
# Standard Library
import os
import sys
import shutil

# Third-party
import testutils

# CAPE
import cape.pyfun.cntl as pfcntl

# Dir to case files
CASEDIR = os.path.join("bullet", "m1.10a0.0b0.0")

# Test Files
TEST_FILES = (
    os.path.join(CASEDIR, "arrow_tec_boundary_timestep200.triq"),
    os.path.join(CASEDIR, "arrow_tec_boundary_timestep200.plt"),
    os.path.join(CASEDIR, "arrow_fm_bullet_total.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "arrow_hist.dat"),
    os.path.join(CASEDIR, "fun3d.00.nml"),
    os.path.join(CASEDIR, "fun3d.01.nml"),
    os.path.join(CASEDIR, "run.01.200"),
    "pyFun.json",
    "matrix.csv",
    "fun3d.nml",
    "functest.py",
    "test.[0-9][0-9].out",
)


# Test Files
TEST_FILES2 = (
    os.path.join(CASEDIR, "arrow_tec_boundary_timestep200.triq"),
    os.path.join(CASEDIR, "arrow_tec_boundary_timestep200.plt"),
    os.path.join(CASEDIR, "arrow_fm_bullet_total.dat"),
    os.path.join(CASEDIR, "case.json"),
    os.path.join(CASEDIR, "arrow_hist.dat"),
    os.path.join(CASEDIR, "fun3d.00.nml"),
    os.path.join(CASEDIR, "fun3d.01.nml"),
    os.path.join(CASEDIR, "run.01.200"),
    "pyFun.json",
    "matrix.csv",
    "fun3d.nml",
    "bullet.xml",
    "functest.py",
    "cap-patch.uh3d",
    "test.[0-9][0-9].out",
    os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds")
)


KW1 = {
    "I": [8],
    "fm": "bullet_total",
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW2 = {
    "I": [8],
    "fm": "bullet_total",
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW3 = {
    "I": [8],
    "ll": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW4 = {
    "I": [8],
    "ll": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW5 = {
    "I": [8],
    "pt": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW6 = {
    "I": [8],
    "pt": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW7 = {
    "I": [8],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW8 = {
    "I": [8],
    "dbpyfunc": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}

KW9 = {
    "I": [8],
    "triqfm": True,
    "restart": True,
    "start": True,
    "__replaced__": []
}

KW10 = {
    "I": [8],
    "triqfm": True,
    "restart": True,
    "start": True,
    "delete": True,
    "__replaced__": []
}


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    cntl = pfcntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW1)
    # Location of output databook
    dbout = os.path.join("data/bullet/aero_bullet_total.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.01.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfm():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.01.out", os.path.join("data", "bullet",
                                            "aero_bullet_total.csv"))
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call FM updater
    cntl.UpdateFM(**KW2)
    # Location of output databook
    dbout = os.path.join("data/bullet/aero_bullet_total.csv")
    # Location of old databook
    dbold = os.path.join("data/bullet/aero_bullet_total.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.02.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_updatedatabookll():
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateLL(**KW3)
    # Location of output databook
    dbout = os.path.join("data/bullet/ll_bullet_total_LL.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.03.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_deletecasesll():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.03.out", os.path.join("data", "bullet",
                                            "ll_bullet_total_LL.csv"))
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateLL(**KW4)
    # Location of output databook
    dbout = os.path.join("data/bullet/ll_bullet_total_LL.csv")
    # Location of old databook
    dbold = os.path.join("data/bullet/ll_bullet_total_LL.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.04.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookpt():
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateTriqPoint(**KW5)
    # Location of output databooks
    dbout1 = os.path.join("data/bullet/pt_p100.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.05.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasespt():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.05.out", os.path.join("data", "bullet", "pt_p100.csv"))
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateTriqPoint(**KW6)
    # Location of output databook
    dbout = os.path.join("data/bullet/pt_p100.csv")
    # Location of old databook
    dbold = os.path.join("data/bullet/pt_p100.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.06.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfunc():
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateDBPyFunc(**KW7)
    # Location of output databooks
    dbout1 = os.path.join("data/bullet/pyfunc_functest.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.07.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfunc():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.07.out",
                os.path.join("data", "bullet", "pyfunc_functest.csv"))
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateDBPyFunc(**KW8)
    # Location of output databook
    dbout = os.path.join("data/bullet/pyfunc_functest.csv")
    # Location of old databook
    dbold = os.path.join("data/bullet/pyfunc_functest.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.08.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_updatedatabooktriqfm():
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateTriqFM(**KW9)
    # Location of output databooks
    dbout1 = os.path.join("data/bullet/triqfm/triqfm_cap.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.09.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES2)
def test_deletecasestriqfm():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    os.mkdir(os.path.join("data", "bullet", "triqfm"))
    # Use test.01.out as existing databook
    shutil.copy("test.09.out",
                os.path.join("data", "bullet", "triqfm", "triqfm_cap.csv"))
    # Get cntl
    cntl = pfcntl.Cntl()
    # Call dbook updater
    cntl.UpdateTriqFM(**KW10)
    # Location of output databook
    dbout = os.path.join("data/bullet/triqfm/triqfm_cap.csv")
    # Location of old databook
    dbold = os.path.join("data/bullet/triqfm/triqfm_cap.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.10.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)
