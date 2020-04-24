#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import numpy as np

# Import DataKit module
import cape.attdb.rdb as rdb


# Read MAT file
db = rdb.DataKit("bullet-fm-mab.mat")

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

# Save figure
h.fig.savefig("python%i-CN-response.png" % sys.version_info.major)

