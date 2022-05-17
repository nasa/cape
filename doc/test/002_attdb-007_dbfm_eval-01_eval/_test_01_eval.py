# -*- coding: utf-8 -*-

# Third-party modules
import testutils

# Import CSV module
import cape.attdb.dbfm as dbfm


# Database name
MAT_FILE = "bullet-fm-mab.mat"
# Test tolerance
TOL = 1e-8


# Test evaluation
@testutils.run_testdir(__file__)
def test_01_linear():
    db = dbfm.DBFM("bullet-fm-mab.mat")
    # FM tags
    fmtags = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
    # Actual FM columns present
    fmcols = []
    # Loop through FM tags
    for tag in fmtags:
        # Extend
        fmcols.append(db.get_col_by_tag(tag))
    # Standard args
    args = ["mach", "alpha", "beta"]
    # Get Mach number break points
    db.create_bkpts(args)
    # Set evaluation
    db.make_responses(fmcols, "linear", args)
    # Pick some conditions
    mach = 0.90
    alph = 1.50
    beta = 0.50
    # Test one evaluation
    assert abs(db("bullet.CN", mach, alph, beta) - 0.0670571) <= TOL
    # Add mapped cols
    qcols = ["q", "T"]
    # Create map of dynamic pressure and temperature vs Mach
    db.create_bkpts_map(qcols, "mach")
    # Set evaluation
    db.make_responses(
        qcols, "linear", ["mach"], response_kwargs={"bkpt": True})
    # Test mapped evaluation
    assert abs(db("T", mach) - 475.33)


# Test moment shifting
@testutils.run_testdir(__file__)
def test_02_clmx():
    # Read DB
    db = dbfm.DBFM("bullet-fm-mab.mat")
    # FM tags
    fmtags = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
    # Actual FM columns present
    fmcols = []
    # Loop through FM tags
    for tag in fmtags:
        # Extend
        fmcols.append(db.get_col_by_tag(tag))
    # Standard args
    args = ["mach", "alpha", "beta"]
    # Get Mach number break points
    db.create_bkpts(args)
    # Set evaluation
    db.make_responses(fmcols, "linear", args)
    # Set evaluator for "bullet.CLMX" and "bullet.CLNX"
    db.make_CLMX()
    db.make_CLNX()
    # Pick some conditions
    mach = 0.90
    alph = 1.50
    beta = 0.50
    xmrp = 2.00
    x = (mach, alph, beta, xmrp)
    # Test CLMX and CLNX
    assert abs(db("bullet.CLMX", *x) - 0.35217470) <= TOL
    assert abs(db("bullet.CLNX", *x) - 0.11741142) <= TOL
        


