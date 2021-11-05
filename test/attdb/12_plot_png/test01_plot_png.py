#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CAPE modules
import cape.attdb.rdb as rdb

# Read a line load datbase
db = rdb.DataKit("bullet-mab.mat")

# Need to set col for x-axis
db.set_output_xargs("bullet.dCN", ["bullet.x"])

# Name of image file to show
fpng = "bullet-xz.png"

# Set a PNG
db.make_png("xz", fpng, ["bullet.dCN"], ImageXMin=-0.15, ImageXMax=4.12)

# Initial plot of a column
h = db.plot("bullet.dCN", 1, XLabel="x/Lref", YLabel="dCN/d(x/Lref)")

# Save figure
h.fig.savefig("python%i-bullet-ll.png" % sys.version_info.major)

