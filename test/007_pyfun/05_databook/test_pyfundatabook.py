
# Standard Library
import os
import sys
import shutil

# Third-party
import testutils

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
    "test.[0-9][0-9].out",
    os.path.join(CASEDIR, "lineload", "LineLoad_bullet_total_LL.dlds"),
)


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookfm():
    # Split update command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--fm",
               "bullet_total"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    print(stdout)
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
    # Split delete command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--fm",
               "bullet_total", "--delete"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
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


@testutils.run_sandbox(__file__, TEST_FILES)
def test_updatedatabookll():
    # Split update command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--ll"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Location of output databook
    dbout = os.path.join("data/bullet/ll_bullet_total_LL.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.03.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesll():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.03.out", os.path.join("data", "bullet",
                                            "ll_bullet_total_LL.csv"))
    # Split delete command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--ll",
               "--delete"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
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
    # Split update command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--pt"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
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
    # Split delete command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--pt",
               "--delete"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
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
    # Split update command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--dbpyfunc"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Location of output databooks
    dbout1 = os.path.join("data/bullet/pyfunc_testfunc.csv")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout1, "test.07.out")
    # Test updated FM Databook
    assert result.line1 == result.line2


@testutils.run_sandbox(__file__, TEST_FILES)
def test_deletecasesfunc():
    os.mkdir("data")
    os.mkdir(os.path.join("data", "bullet"))
    # Use test.01.out as existing databook
    shutil.copy("test.05.out",
                os.path.join("data", "bullet","pyfunc_testfunc.csv"))
    # Split delete command and add `-m` prefix
    cmdlist = [sys.executable, "-m", "cape.pyfun", "-I", "8", "--dbpyfunc",
               "--delete"]
    # Run command
    stdout, _, _ = testutils.call_o(cmdlist)
    # Location of output databook
    dbout = os.path.join("data/pyfunc_testfunc.csv")
    # Location of old databook
    dbold = os.path.join("data/pyfunc_testfunc.csv.old")
    # Compare output databook with reference result
    result = testutils.compare_files(dbout, "test.08.out")
    # Test deleted FM Databook
    assert result.line1 == result.line2
    # Test old databook exists
    assert os.path.exists(dbold)
