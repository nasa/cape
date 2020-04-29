#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library modules
import sys

# Third-party modules
import numpy as np

# Import CAPE modules
import cape.tnakit.plot_mpl as pmpl


# Circle for sample data
th = np.linspace(0, 2*np.pi, 61)
x = np.cos(th)
y = np.sin(th)

# Create plot
h = pmpl.plot(x, y, TopSpine=True, RightSpine=True)

# Start making labels
pmpl.axlabel("Position 8", pos=8, color="purple")
pmpl.axlabel("Position 15", pos=15, rotation=-90)

# Make labels using automatic positions
pmpl.axlabel(u"μ = 0.0000", color="red")
pmpl.axlabel(u"r = 1.0000", color="navy", fontdict={"weight": "bold"})
pmpl.axlabel("Third auto position")

# Redo margins
pmpl.axes_adjust(h.fig)

# Save figure
h.fig.savefig("python%i-axlabel.png" % sys.version_info.major)

