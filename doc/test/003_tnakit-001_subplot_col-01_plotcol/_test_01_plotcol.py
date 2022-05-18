#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import os.path as op
import sys

# Third-party modules
import testutils

# Local imports
import cape.attdb.rdb as rdb
import cape.tnakit.plot_mpl as pmpl


# File names
FDIR = op.dirname(__file__)
MATFILE = "bullet-mab.mat"
PNGFILE = "bullet-xz.png"
TESTFILES = (MATFILE, PNGFILE)


# Test subplots of columns
@testutils.run_sandbox(__file__, TESTFILES)
def test_01_subplot_bullet_ll():
    # Read a line load datbase
    db = rdb.DataKit(MATFILE)
    # Initial plot of a column
    h = pmpl.plot(
        db["bullet.x"], db["bullet.dCN"][:, 1], YLabel="dCN/d(x/Lref)")
    # Add an axes
    ax = h.fig.add_subplot(212)
    # Plot the image
    pmpl.imshow(PNGFILE, ImageXMin=-0.15, ImageXMax=4.12)
    # Format nicely
    pmpl.axes_adjust_col(h.fig, SubplotRubber=1)
    # Tie horizontal limits
    ax.set_xlim(h.ax.get_xlim())
    # Image file name
    fimg = "python%i-bullet-ll.png" % sys.version_info.major
    # Save figure
    h.fig.savefig(fimg)
    h.close()
    # Compare figure
    testutils.assert_png(fimg, op.join(FDIR, fimg))

