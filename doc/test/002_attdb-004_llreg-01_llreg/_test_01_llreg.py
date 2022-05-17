# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np
import testutils

# Import CSV module
import cape.attdb.rdb as rdb


# Regularize some line loads
@testutils.run_testdir(__file__)
def test_01_llgriddata():
    db = rdb.DataKit("bullet-mab.mat")
    # Number of break points
    n = 17
    # Reference points for regularization
    A0 = np.linspace(-2, 2, n)
    B0 = np.linspace(-2, 2, n)
    # Save break points
    db.bkpts = {"alpha": A0, "beta": B0, "mach": np.unique(db["mach"])}
    # Regularize
    db.regularize_by_griddata(
        "bullet.dCN",
        ["alpha", "beta"],
        scol="mach",
        prefix="reg.")
    # Check shape
    assert db["reg.bullet.dCN"].shape == (51, 2*n*n)
