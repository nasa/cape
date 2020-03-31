#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import numpy as np

# CAPE modules
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-reg.mat")

# Test cols
cols = ["bullet.CN", "bullet.dCN"]

# Standard args
args = ["mach", "alpha", "beta"]
# Get Mach number break points
db.create_bkpts(args)

# Set evaluation
db.make_responses(cols, "linear", args)
db.set_output_xargs("bullet.dCN", ["bullet.x"])

# Pick some conditions
mach = 0.95
alph = 1.50
beta = 0.00

# Find those conditions
I, _ = db.find(args, mach, alph, beta)
# Unpack
i, = I

# Plot it with all defaults
h = db.plot("bullet.dCN", i)

# Save result
h.fig.savefig("python%i-dCN-index.png" % sys.version_info.major, dpi=90)

# Show handle
print("Index: %i" % i)

