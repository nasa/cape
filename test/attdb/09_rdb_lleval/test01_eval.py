#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library


# Third-party modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import CSV module
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-ll-reg.mat")

# Standard args
args = ["mach", "alpha", "beta"]
# Get Mach number break points
db.create_bkpts(args)

# Set evaluation
db.make_response("bullet.dCN", "linear", args)
# Set output args
db.set_output_xargs("bullet.dCN", ["bullet.x"])

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50

# Conditions
print("mach: %.2f" % mach)
print("alpha: %.2f" % alph)
print("beta: %.2f" % beta)
# Evaluate line load
dCN = db("bullet.dCN", mach, alph, beta)
# Make sure it can be plotted
plt.plot(db["bullet.x"], dCN)
# Display size
print("bullet.dCN.size: %i" % dCN.size)
# Display output args
print("bullet.dCN.xargs: %s" % db.get_output_xargs("bullet.dCN"))

