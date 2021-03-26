#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.plt

# Output tri file
PLTFILE = "arrow-tri10k.plt"

# Read triangulation output from test01
plt = cape.plt.Plt(fname=PLTFILE)
