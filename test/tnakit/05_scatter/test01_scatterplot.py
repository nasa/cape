#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library modules
import sys

# Third-party modules
import numpy as np

# Import CAPE modules
import cape.attdb.dbfm as dbfm
import cape.tnakit.plot_mpl as pmpl


# Read CSV file
db = dbfm.DBFM("bullet-fm.mat")

# Name of image file to show
fpng = "bullet-deltaCA-scatter.png"

# Creat random CA offset
np.random.seed(100)
dCA = 0.1 * np.random.rand(len(db["bullet.CA"]))

# Set colormap keyword
kw = {"ScatterColor": db["alpha"], "XLabel": "beta", "YLabel": "delCA"}

# Plot contour
h = pmpl.scatter(db["beta"], db["bullet.CA"]-dCA, **kw)

# Save figure
h.fig.savefig("python%i-bullet-deltaCA-scatter.png" % sys.version_info.major)

