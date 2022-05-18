# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.csvfile as csvfile


# File name
CSVFILE = "wt-sample.csv"


# Test integer read
@testutils.run_testdir(__file__)
def test_01_clean():
    # Read CSV file
    db = csvfile.CSVFile(CSVFILE)
    # Case number
    i = 2
    # Get attributes
    run = db["run"][i]
    pt = db["pt"][i]
    # Test
    assert run == 257
    assert pt == 3


# Test data types w/ kwarg
@testutils.run_testdir(__file__)
def test_02_dtypes():
    # Read CSV file
    db = csvfile.CSVFile("wt-sample.csv", DefaultType="int32")
    # Test types
    assert db["mach"].dtype.name == "float64"
    assert db["run"].dtype.name == "int32"
    assert db["pt"].dtype.name == "int32"

