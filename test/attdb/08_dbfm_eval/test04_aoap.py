#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# ATTDB modules
import cape.attdb.convert as convert
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

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50
# Convert
aoap, phip = convert.AlphaBeta2AlphaTPhi(alph, beta)

# Conditions
print("mach: %.2f" % mach)
print("aoa : %.2f" % alph)
print("beta: %.2f" % beta)
print("aoap: %.4f" % aoap)
print("phip: %.4f" % phip)
# Loop throuch columns
for col in fmcols:
    # Evaluate using *alpha*, *beta*
    v1 = db(col, mach, alph, beta)
    # Evaluate using *aoa*, *BETA*
    v2 = db(col, mach, aoa=alph, BETA=beta)
    # Evaluate using *aoap*, *phip*
    v3 = db(col, mach, aoap=aoap, phip=phip)
    # Print it
    print("%-10s: %.3f %.3f %.3f" % (col, v1, v2, v3))

# Get *aoap*, *phip* cols
db.make_aoap_phip()

# Print sizes
for col in ["aoap", "phip"]:
    # Get tag
    dtype = db.get_col_dtype(col)
    print("%s: size=%s, dtype=%s" % (col, db[col].size, dtype))

