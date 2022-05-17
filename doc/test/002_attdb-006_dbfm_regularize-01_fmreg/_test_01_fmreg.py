# -*- coding: utf-8 -*-

# Standard library modules
import sys

# Third-party modules
import numpy as np
import testutils

# Import CSV module
import cape.attdb.dbfm as dbfm


# Database file
MAT_FILE = "bullet-fm.mat"


# Griddata regularize
@testutils.run_testdir(__file__)
def test_01_griddata():
    db = dbfm.DBFM("bullet-fm.mat")
    # Actual FM columns present
    fmcols = []
    # Loop through FM tags
    for tag in ["CA", "CY", "CN", "CLL", "CLM", "CLN"]:
        # Extend
        fmcols.append(db.get_col_by_tag(tag))
    # Number of break points
    n = 17
    # Reference points for regularization
    A0 = np.linspace(-2, 2, n)
    B0 = np.linspace(-2, 2, n)
    # Save break points
    db.bkpts = {"alpha": A0, "beta": B0}
    # Get Mach number break points
    db.create_bkpts(["mach", "q", "T"])
    # Regularize
    db.regularize_by_griddata(
        fmcols, ["alpha", "beta"], prefix="reg.", scol="mach")
    # Check number of Mach numbers
    nmach = db.bkpts["mach"].size
    # Check regularlized columns
    for col in ("bullet.CA", "bullet.CLM", "q", "T", "mach"):
        # Form column name
        regcol = "reg." + col
        # Check present and its size
        assert regcol in db
        assert db[regcol].size == nmach * n * n


# Griddata regularize
@testutils.run_testdir(__file__)
def test_02_rbf():
    db = dbfm.DBFM("bullet-fm.mat")
    # Actual FM columns present
    fmcols = []
    # Loop through FM tags
    for tag in ["CA", "CY", "CN", "CLL", "CLM", "CLN"]:
        # Extend
        fmcols.append(db.get_col_by_tag(tag))
    # Number of break points
    n = 17
    # Reference points for regularization
    A0 = np.linspace(-2, 2, n)
    B0 = np.linspace(-2, 2, n)
    # Save break points
    db.bkpts = {"alpha": A0, "beta": B0}
    # Get Mach number break points
    db.create_bkpts(["mach", "q", "T"])
    # Regularize
    db.regularize_by_rbf(
        fmcols, ["alpha", "beta"], prefix="reg.", scol="mach")
    # Check number of Mach numbers
    nmach = db.bkpts["mach"].size
    # Check regularlized columns
    for col in ("bullet.CA", "bullet.CLM", "q", "T", "mach"):
        # Form column name
        regcol = "reg." + col
        # Check present and its size
        assert regcol in db
        assert db[regcol].size == nmach * n * n
    
