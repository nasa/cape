#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import matplotlib.pyplot as plt

# CAPE modules
import cape.attdb.rdb as rdb
import cape.tnakit.plot_mpl as pmpl

# Read a line load datbase
db = rdb.DataKit("bullet-mab.mat")

# Name of image file to show
fpng = "bullet-xz.png"

# Set a PNG
db.make_png("xz", fpng, ["bullet.dCN"], ImageXMin=-0.15, ImageXMax=4.12)

# Initial plot of a column
h = pmpl.plot(db["bullet.x"], db["bullet.dCN"][:,1], YLabel="dCN/d(x/Lref)")

# Plot the image
h = db.plot_png("bullet.dCN", h.fig, h=h)

# Save figure
h.fig.savefig("python%i-bullet-ll.png" % sys.version_info.major)

