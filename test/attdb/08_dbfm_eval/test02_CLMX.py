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
db.create_bkpts(args)

# Set evaluation
db.make_responses(fmcols, "linear", args)
# Set evaluator for "bullet.CLMX" and "bullet.CLNX"
db.make_CLMX()
db.make_CLNX()

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50
xmrp = 2.00

# Conditions
print("mach : %.2f" % mach)
print("alpha: %.2f" % alph)
print("beta : %.2f" % beta)
print("xMRP : %.2f" % xmrp)

# Loop throuch columns
for col in ["CLM", "CLN"]:
    # Column names
    col1 = "bullet.%s" % col
    col2 = "bullet.%sX" % col
    # Evaluate
    v1 = db(col1, mach, alph, beta)
    v2 = db(col2, mach, alph, beta, xmrp)
    # Print them
    print("%-11s: %.3f" % (col1, v1))
    print("%-11s: %.3f" % (col2, v2))

