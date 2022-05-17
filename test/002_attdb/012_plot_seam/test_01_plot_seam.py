# -*- coding: utf-8 -*-

# Standard library
import os
import sys

# Third-party modules
import testutils

# Local imports
import cape.attdb.rdb as rdb


# This folder
FDIR = os.path.dirname(__file__)
MAT_FILE = "bullet-mab.mat"
SMY_FILE = "arrow.smy"
TEST_FILES = (MAT_FILE, SMY_FILE)


# Test a line load plot with seam curves
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_plot_seam():
    # Read a line load datbase
    db = rdb.DataKit(MAT_FILE)
    # Need to set col for x-axis
    db.set_output_xargs("bullet.dCN", ["bullet.x"])
    # Name of seam curve file
    fseam = SMY_FILE
    # Seam title
    seam = "smy"
    # Seam col names
    xcol = "smy.x"
    ycol = "smy.z"
    # Cols for this seam curve
    cols = ["bullet.dCN"]
    # Set up a seam curve
    db.make_seam(seam, fseam, xcol, ycol, cols)
    # Divide seam curves by two (what happened?)
    db["smy.x"] *= 0.5
    db["smy.z"] *= 0.5
    # Initial plot of a column
    h = db.plot("bullet.dCN", 1, XLabel="x/Lref", YLabel="dCN/d(x/Lref)")
    # Image file name
    fimg = "python%i-bullet-ll.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Compare files
    testutils.assert_png(fimg, os.path.join(FDIR, fimg), tol=0.93)


if __name__ == "__main__":
    test_01_plot_seam()
