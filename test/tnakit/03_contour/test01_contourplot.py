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
fpng = "bullet-CLM-contour.png"

# Get cases for plotting
i = np.where(db['mach']==0.8)[0]

# Set colormap keyword
kw = {"ContourColorMap": "RdBu", "XLabel": "Alpha", "YLabel": "Beta"}

# Plot contour
h = pmpl.contour(db["alpha"][i], db["beta"][i], db["bullet.CLM"][i], **kw)

# Save figure
h.fig.savefig("python%i-bullet-CLM-contour.png" % sys.version_info.major)

