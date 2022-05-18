# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np
import testutils

# Import CSV module
import cape.attdb.ftypes.xlsfile as xlsfile


# File name
XLSFILE = "header_categories.xlsx"


# Test multiple worksheets and 2D "col"
@testutils.run_testdir(__file__)
def test_01_workbook():
    # Read CSV file
    db = xlsfile.XLSFile(XLSFILE)
    # Test sizes and types
    assert "colnames.mach" in db
    assert "colnames.config" in db
    assert "cols_with_array.beta" in db
    assert "cols_with_array.DCN" in db
    assert db["colnames.mach"].dtype.name == "float64"
    assert db["colnames.mach"].shape == (75,)
    assert isinstance(db["colnames.config"], list)
    assert db["cols_with_array.beta"].shape == (3,)
    assert db["cols_with_array.DCN"].shape == (3, 3)

