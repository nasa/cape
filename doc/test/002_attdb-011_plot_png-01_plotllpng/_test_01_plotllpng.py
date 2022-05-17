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
# Data file
MAT_FILE = "bullet-mab.mat"
PNG_FILE = "bullet-xz.png"
TEST_FILES = (MAT_FILE, PNG_FILE)


# Test line load with PNG for seam
@testutils.run_sandbox(__file__, TEST_FILES)
def test_01_plot_pngseam():
    # Read a line load datbase
    db = rdb.DataKit(MAT_FILE)
    # Need to set col for x-axis
    db.set_output_xargs("bullet.dCN", ["bullet.x"])
    # Name of image file to show
    fpng = PNG_FILE
    # Set a PNG
    db.make_png("xz", fpng, ["bullet.dCN"], ImageXMin=-0.15, ImageXMax=4.12)
    # Initial plot of a column
    h = db.plot("bullet.dCN", 1, XLabel="x/Lref", YLabel="dCN/d(x/Lref)")
    # PNG file name
    fimg = "python%i-bullet-ll.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    # Test it
    testutils.assert_png(fimg, os.path.join(FDIR, fimg), tol=0.93)

