
# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.xlsfile as xlsfile


# File names
XLSFILE = "aero_arrow_no_base.xls"
XLSXFILE = XLSFILE + "x"
# Test tolerance
TOL = 1e-8


# Test a basic read
@testutils.run_testdir(__file__)
def test_01_xlsx():
    # Read CSV file
    db = xlsfile.XLSFile(XLSXFILE)
    # Case number
    i = 6
    # Test values and types
    assert isinstance(db["config"], list)
    assert db["config"][i] == "poweroff"
    assert abs(db["CA"][i] - 0.34490558) <= TOL
    assert abs(db["nStats"][i] - 100) <= TOL


# Test a basic read, older format
@testutils.run_testdir(__file__)
def test_02_xls():
    # Read CSV file
    db = xlsfile.XLSFile(XLSFILE)
    # Case number
    i = 6
    # Test values and types
    assert isinstance(db["config"], list)
    assert db["config"][i] == "poweroff"
    assert abs(db["CA"][i] - 0.34490558) <= TOL
    assert abs(db["nStats"][i] - 100) <= TOL


# Test data types
@testutils.run_testdir(__file__)
def test_03_dtypes():
    # Read CSV file
    db = xlsfile.XLSFile(
        XLSXFILE,
        Types={
            "config": "str",
            "alpha": "float16",
            "mach": "float32",
            "nStats": "int"
        })
    # Test data types
    assert db["mach"].dtype.name == "float32"
    assert db["alpha"].dtype.name == "float16"
    assert db["CLM"].dtype.name == "float64"
    assert db["nStats"].dtype.name == "int32"
