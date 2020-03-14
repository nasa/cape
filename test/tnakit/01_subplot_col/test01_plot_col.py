#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import matplotlib.pyplot as plt

# CAPE modules
import cape.attdb.rdb as rdb
import cape.tnakit.plot_mpl as pmpl

# Read a line load datbase
db = rdb.DataKit("bullet-mab.mat")

# Initial plot of a column
h = pmpl.plot(db["bullet.x"], db["bullet.dCN"][:,1], YLabel="dCN/d(x/Lref)")

# Add an axes
ax = h.fig.add_subplot(212)

# Name of image file to show
fpng = "bullet-xz.png"

# Plot the image
pmpl.imshow(fpng, ImageXMin=-0.15, ImageXMax=4.12)

# Format nicely
pmpl.axes_adjust_col(h.fig, SubplotRubber=1)

# Tie horizontal limits
ax.set_xlim(h.ax.get_xlim())

# Save figure
h.fig.savefig("bullet-ll.png")

