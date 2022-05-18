# -*- coding: utf-8 -*-

# Third-pary
import testutils

# Local imports
import cape.attdb.ftypes.matfile as matfile


# File names
MATFILE = "bullet.mat"
MATFILE1 = "bullet1.mat"


# Test 2D shape
@testutils.run_sandbox(__file__, MATFILE)
def test_01_matshape():
    # Read MAT file
    db = matfile.MATFile(MATFILE)
    # Expected shapes
    ncase = 6
    nx = 101
    # Check shapes
    assert db["T"].size == ncase
    assert db["bullet.x"].size == nx
    assert db["bullet.dCN"].shape == (nx, ncase)


# Test write
@testutils.run_sandbox(__file__, fresh=False)
def test_02_matwrite():
    # Read MAT file
    db = matfile.MATFile(MATFILE)
    # Rename a column
    db["MACH"] = db.pop("mach")
    db.cols.remove("mach")
    db.cols.append("MACH")
    # Write it
    db.write_mat(MATFILE1)
    # Reread
    db = matfile.MATFile(MATFILE1)
    # Expected shapes
    ncase = 6
    nx = 101
    # Check shapes
    assert db["T"].size == ncase
    assert db["bullet.x"].size == nx
    assert db["bullet.dCN"].shape == (nx, ncase)

