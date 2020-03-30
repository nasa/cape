#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
from cape.tnakit.plot_mpl import MPLOpts, rstutils

# Try *PlotLineStyle*
opts = MPLOpts(PlotLineStyle="--", Index=2)
# Get "PlotOptions" for case 3
plot_opts = opts.get_option("PlotOptions")

# Show options for plot()
print("PlotOptions:")
print(rstutils.py2rst(plot_opts, indent=4))

