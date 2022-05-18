# -*- coding: utf-8 -*-

# Third-party modules
import testutils

# Local imports
import cape.attdb.dbfm as dbfm


# Test LL evaluation
@testutils.run_testdir(__file__)
def test_01_ll_eval():
    db = dbfm.DBFM("bullet-ll-reg.mat")
    # Standard args
    args = ["mach", "alpha", "beta"]
    # Get Mach number break points
    db.create_bkpts(args)
    # Set evaluation
    db.make_response("bullet.dCN", "linear", args)
    # Set output args
    db.set_output_xargs("bullet.dCN", ["bullet.x"])
    # Pick some conditions
    mach = 0.90
    alph = 1.50
    beta = 0.50
    # Evaluate line load (functionality test mostly)
    dCN = db("bullet.dCN", mach, alph, beta)
    # Check size
    assert dCN.size == db["bullet.x"].size
    # Display output args
    assert db.get_output_xargs("bullet.dCN") == ["bullet.x"]
    assert db.get_output_xarg1("bullet.dCN") == "bullet.x"
