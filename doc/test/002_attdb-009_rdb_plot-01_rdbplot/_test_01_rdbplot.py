# -*- coding: utf-8 -*-

# Standard library
import os
import sys

# Third-party modules
import numpy as np
import testutils

# Import CSV module
import cape.attdb.dbfm as dbfm


# Data file
MAT_FILE = "bullet-reg.mat"
# Test tolerance
TOL = 1e-6


# Test preprocessing
@testutils.run_testdir(__file__)
def test_01_preplot():
    # Read MAT file
    db = dbfm.DBFM(MAT_FILE)
    # Test cols
    cols = ["bullet.CN", "bullet.dCN"]
    # Standard args
    args = ["mach", "alpha", "beta"]
    # Get Mach number break points
    db.create_bkpts(args)
    # Set evaluation
    db.make_responses(cols, "linear", args)
    db.set_output_xargs("bullet.dCN", ["bullet.x"])
    # Pick some conditions
    mach = 0.90
    alph = 1.50
    beta = 0.50
    i1 = 200
    i2 = 238
    i3 = 240
    i4 = 251
    # Preprpocess
    args1 = db._prep_args_plot1("bullet.CN", mach, alph, beta)
    args2 = db._prep_args_plot1("bullet.CN", np.array([i1, i2]))
    args3 = db._prep_args_plot1("bullet.dCN", [i3, i4])
    # Unpack
    mach1, aoa1, beta1 = args1[3]
    mach2, aoa2, beta2 = args2[3]
    mach3, aoa3, beta3 = args3[3]
    # Test
    assert abs(mach1 - mach) <= TOL
    assert abs(aoa1 - alph) <= TOL
    assert abs(beta1 - beta) <= TOL
    assert abs(mach2[0] - db["mach"][i1]) <= TOL
    assert abs(aoa2[0] - db["alpha"][i1]) <= TOL
    assert abs(beta2[1] - db["beta"][i2]) <= TOL
    assert abs(mach3[0] - db["mach"][i3]) <= TOL
    assert abs(aoa3[0] - db["alpha"][i3]) <= TOL
    assert abs(beta3[1] - db["beta"][i4]) <= TOL


# Read CSV file
@testutils.run_sandbox(__file__, MAT_FILE)
def test_02_plot_ll():
    db = dbfm.DBFM(MAT_FILE)
    # Test cols
    cols = ["bullet.CN", "bullet.dCN"]
    # Standard args
    args = ["mach", "alpha", "beta"]
    # Get Mach number break points
    db.create_bkpts(args)
    # Set evaluation
    db.make_responses(cols, "linear", args)
    db.set_output_xargs("bullet.dCN", ["bullet.x"])
    # Pick some conditions
    mach = 0.95
    alph = 1.50
    beta = 0.00
    # Find those conditions
    I, _ = db.find(args, mach, alph, beta)
    # Unpack
    i, = I
    assert i == 535
    # Plot it with all defaults
    h = db.plot("bullet.dCN", i)
    # File name (slightly different Python 2 vs 3)
    fpng = "python%i-dCN-index.png" % sys.version_info.major
    fabs = os.path.abspath(fpng)
    fdir = os.path.dirname(os.getcwd())
    ftarg = os.path.join(fdir, fpng)
    # Save result
    h.fig.savefig(fpng, dpi=90)
    h.close()
    # Compare image
    assert testutils.assert_png(fabs, ftarg, tol=0.93)

