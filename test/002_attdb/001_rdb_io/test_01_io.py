# -*- coding: utf-8 -*-

# Third party
import testutils

# Local imports
import cape.attdb.rdb as rdb


# Files to copy
CSV_FILE = "aero_arrow_no_base.csv"


# Read CSV and convert to MAT
@testutils.run_sandbox(__file__, CSV_FILE)
def test_01_csv2mat():
    db = rdb.DataKit("aero_arrow_no_base.csv")
    # Case number
    i = 13
    # Get attributes
    mach = db["mach"][i]
    alph = db["alpha"][i]
    CA = db["CA"][i]
    # Test
    assert abs(mach - 0.95) <= 1e-8
    assert abs(alph - 5.00) <= 1e-8
    assert abs(CA - 0.52628) <= 1e-4
    # Write MAT file
    db.write_mat("aero_arrow_no_base.mat")
    # Reread
    db1 = rdb.DataKit("aero_arrow_no_base.mat")
    # Get attributes
    mach = db1["mach"][i]
    alph = db1["alpha"][i]
    CA = db1["CA"][i]
    # Test
    assert abs(mach - 0.95) <= 1e-8
    assert abs(alph - 5.00) <= 1e-8
    assert abs(CA - 0.52628) <= 1e-4

