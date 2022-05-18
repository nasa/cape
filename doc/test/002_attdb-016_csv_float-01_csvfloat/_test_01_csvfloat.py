# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.csvfile as csvfile


# Files
CSVFILE = "aeroenv.csv"
# Tolerance for tests
TOL = 1e-8


# Test read
@testutils.run_testdir(__file__)
def test_01_csvfloat():
    # Read CSV file
    db = csvfile.CSVFile(CSVFILE)
    # Case number
    i = 13
    # Get attributes
    mach = db["mach"][i]
    beta = db["beta"][i]
    # Test values
    assert abs(mach - 1.2) <= TOL
    assert abs(beta - 4.0) <= TOL


# Test data types
@testutils.run_testdir(__file__)
def test_02_csvdtype():
    # Read CSV file (w/ options0
    db = csvfile.CSVFile(
        CSVFILE,
        DefaultType="float32",
        Definitions={"beta": {"Type": "float16"}})
    # Print data types
    assert db.cols == ["mach", "alpha", "beta"]
    assert db["mach"].dtype.name == "float32"
    assert db["beta"].dtype.name == "float16"


# Test "simple" CSV (faster)
@testutils.run_testdir(__file__)
def test_03_csvsimple():
    # Read CSV file
    db = csvfile.CSVSimple(CSVFILE)
    # Case number
    i = 13
    # Get attributes
    mach = db["mach"][i]
    beta = db["beta"][i]
    # Test values
    assert abs(mach - 1.2) <= TOL
    assert abs(beta - 4.0) <= TOL


# Test extension read
@testutils.run_testdir(__file__)
def test_04_csv_c():
    # Read CSV file
    db = csvfile.CSVFile()
    # Read in C
    db.c_read_csv(CSVFILE)
    # Case number
    i = 13
    # Get attributes
    mach = db["mach"][i]
    beta = db["beta"][i]
    # Test values
    assert abs(mach - 1.2) <= TOL
    assert abs(beta - 4.0) <= TOL


# Test explicit pytho nread
@testutils.run_testdir(__file__)
def test_05_csv_py():
    # Read CSV file
    db = csvfile.CSVFile()
    # Read in C
    db.py_read_csv(CSVFILE)
    # Case number
    i = 13
    # Get attributes
    mach = db["mach"][i]
    beta = db["beta"][i]
    # Test values
    assert abs(mach - 1.2) <= TOL
    assert abs(beta - 4.0) <= TOL

