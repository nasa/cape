#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os.path as op
import sys

# Third-party modules
import testutils

# Import DataKit module
import cape.attdb.rdb as rdb


# This folder
FDIR = op.dirname(__file__)
MAT_FILE = "bullet-fm-mab.mat"


# Test a contour plot
@testutils.run_sandbox(__file__, MAT_FILE)
def test_01_plot_contour():
    # Read MAT file
    db = rdb.DataKit(MAT_FILE)
    # Column to plot
    col = "bullet.CN"
    # Filter
    I, _ = db.find(["mach"], 0.95)
    # Other options
    kw = {
        "ContourColorMap": "RdYlBu_r",
        "MarkerSize": 3,
    }
    # Plot
    h = db.plot_contour(col, I, xk="beta", yk="alpha", **kw)
    # Image file name
    fimg = "python%i-CN-mask.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Test it
    testutils.assert_png(fimg, op.join(FDIR, fimg), tol=0.93)


# Test customizaing contour levels
@testutils.run_sandbox(__file__, fresh=False)
def test_02_contour_levels():
    # Read MAT file
    db = rdb.DataKit(MAT_FILE)
    # Column to plot
    col = "bullet.CN"
    # Filter
    I, _ = db.find(["mach"], 0.95)
    # Other options
    kw = {
        "ContourColorMap": "bwr",
        "ContourLevels": 20,
        "MarkerOptions": {
            "marker": "v",
            "markerfacecolor": "none",
            "markersize": 4,
        },
    }
    # Plot
    h = db.plot_contour(col, I, xk="beta", yk="alpha", **kw)
    # Image file name
    fimg = "python%i-CN-levels.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Test it
    testutils.assert_png(fimg, op.join(FDIR, fimg), tol=0.93)


@testutils.run_sandbox(__file__, fresh=False)
def test_03_response_contour():
    # Read MAT file
    db = rdb.DataKit(MAT_FILE)
    # Column to plot
    col = "bullet.CN"
    # Args
    args = ["mach", "alpha", "beta"]
    # Create response
    db.make_response(col, "linear", args)
    # Create break points
    db.create_bkpts(args)
    # Filter
    I, _ = db.find(["mach"], 0.95)
    # Get mach, alpha, beta values
    mach = 0.90
    alpha = db["alpha"][I]
    beta = db["beta"][I]
    # Other options
    kw = {
        "ContourColorMap": "RdYlBu_r",
    }
    # Plot
    h = db.plot_contour(col, mach, alpha, beta, xk="beta", yk="alpha", **kw)
    # Image file name
    fimg = "python%i-CN-response.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Test it
    testutils.assert_png(fimg, op.join(FDIR, fimg), tol=0.93)

