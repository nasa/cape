#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import numpy as np
import testutils

# Local imports
import cape.attdb.rdb as rdb


MAT_FILE = "bullet-mab.mat"
TOL = 1e-6


# Integrate some line loads
@testutils.run_sandbox(__file__, MAT_FILE)
def test_01_trapz():
    db = rdb.DataKit(MAT_FILE)
    # Get key fields
    x = db["bullet.x"]
    dCY = db["bullet.dCY"]
    dCN = db["bullet.dCN"]
    # MRP
    xmrp = np.max(x)
    # xhat for moments
    xhat = xmrp - x
    # Initialize moments
    dCLL = np.zeros_like(dCN)
    dCLM = np.zeros_like(dCN)
    dCLN = np.zeros_like(dCY)
    # Dimensions
    _, n = dCY.shape
    # Create moments for each condition
    for i in range(n):
        # Create pitching and yawing moments
        dCLM[:, i] = xhat * dCN[:, i]
        dCLN[:, i] = xhat * dCY[:, i]
    # Save moments
    db.save_col("bullet.dCLL", dCLL)
    db.save_col("bullet.dCLM", dCLM)
    db.save_col("bullet.dCLN", dCLN)
    # Save definitions
    db.make_defn("bullet.dCLL", dCLL)
    db.make_defn("bullet.dCLM", dCLM)
    db.make_defn("bullet.dCLN", dCLN)
    # Integrate all fields
    for col in ["CA", "CY", "CN", "CLL", "CLM", "CLN"]:
        # Integrate
        db.create_integral("bullet.d%s" % col, "bullet.x")
    # Save it
    db.write_mat("bullet-FM-LL.mat")
    # Test columns
    assert sorted(db.cols) == [
        "T",
        "alpha",
        "aoap",
        "beta",
        "bullet.CA",
        "bullet.CLL",
        "bullet.CLM",
        "bullet.CLN",
        "bullet.CN",
        "bullet.CY",
        "bullet.dCA",
        "bullet.dCLL",
        "bullet.dCLM",
        "bullet.dCLN",
        "bullet.dCN",
        "bullet.dCY",
        "bullet.x",
        "mach",
        "phip",
        "q"
    ]
    # Pick a case
    i = 9
    # Check some values
    assert abs(db["mach"][i] - 0.8) <= TOL
    assert abs(db["alpha"][i] - 2.0) <= TOL
    assert abs(db["bullet.CN"][i] - 0.092838) <= TOL
    assert abs(db["bullet.CLM"][i] - 0.2871063) <= TOL
