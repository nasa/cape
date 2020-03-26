#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-fm-mab.mat")

# FM tags
fmtags = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
# Actual FM columns present
fmcols = []
# Loop through FM tags
for tag in fmtags:
    # Extend
    fmcols.append(db.get_col_by_tag(tag))

# Standard args
args = ["mach", "alpha", "beta"]
# Get Mach number break points
db.get_bkpts(args)

# Set evaluation
db.make_responses(fmcols, "linear", args)
# Set evaluator for "bullet.CLMX"
db.make_CLMX()

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50

print("CLMX: %.2f" % db("bullet.CLMX", mach, alph, beta, xMRP=1.0))

# Conditions
print("mach: %.2f" % mach)
print("alpha: %.2f" % alph)
print("beta: %.2f" % beta)
# Loop throuch columns
for col in fmcols:
    # Print it
    print("%s: %.2f" % (col, db(col, mach, alph, beta)))

