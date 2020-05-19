#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import matplotlib.pyplot as plt

# CAPE modules
import cape.attdb.rdb as rdb

# Read a line load datbase
db = rdb.DataKit("bullet-mab.mat")

# Need to set col for x-axis
db.set_output_xargs("bullet.dCN", ["bullet.x"])

# Name of seam curve file
fseam = "arrow.smy"
# Seam title
seam = "smy"
# Seam col names
xcol = "smy.x"
ycol = "smy.z"
# Cols for this seam curve
cols = ["bullet.dCN"]

# Set up a seam curve
db.make_seam(seam, fseam, xcol, ycol, cols)

# Initial plot of a column
h = db.plot("bullet.dCN", 1, XLabel="x/Lref", YLabel="dCN/d(x/Lref)")

# Save figure
h.fig.savefig("python%i-bullet-ll.png" % sys.version_info.major)

