# -*- coding: utf-8 -*-

# Third-party
import testutils

# Local imports
import cape.attdb.ftypes.matfile as matfile


# File names
MATFILE = "wt-sample.mat"


# Test basic MAT file read
@testutils.run_testdir(__file__)
def test_01_matread():
    # Read MAT file
    db = matfile.MATFile(MATFILE)
    # Case number
    i = 3
    # Test
    assert db["run"][i] == 257
    assert db["pt"][i] == 4
    assert db["run"].dtype.name.startswith("int")
    assert db["mach"].dtype.name.startswith("float")

