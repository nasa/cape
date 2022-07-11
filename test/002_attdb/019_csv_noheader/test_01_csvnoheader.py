# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.csvfile as csvfile


# File names
CSVFILE = "runmatrix.csv"
# Test tolerance
TOL = 1e-8


# Test mixed-format auto read
@testutils.run_testdir(__file__)
def test_01_csvmixed():
    # Read CSV file
    db = csvfile.CSVFile(CSVFILE)
    # Case number
    i = 6
    # Test values
    assert abs(db["col1"][i] - 2.1) <= TOL
    assert db["col4"][i] == "poweroff"
    assert db["col5"][i] == ""
    assert db["col6"][i] == "@user3"


# Test data types in same
@testutils.run_testdir(__file__)
def test_02_dtypes():
    # Read CSV file w/ some options
    db = csvfile.CSVFile(
        CSVFILE,
        cols=["mach", "alpha", "beta", "config", "Label", "user"],
        DefaultType="float32",
        Types={
            "config": "str",
            "mach": "float16"
        })
    # Test data types
    assert isinstance(db["config"], list)
    assert isinstance(db["config"][0], str)
    assert db["mach"].dtype.name == "float16"
    assert db["alpha"].dtype.name == "float32"

