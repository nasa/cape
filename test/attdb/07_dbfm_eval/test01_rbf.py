#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-fm.mat")

# Actual FM columns present
fmcols = []
# Loop through FM tags
for tag in ["CA", "CY", "CN", "CLL", "CLM", "CLN"]:
    # Extend
    fmcols.append(db.get_col_by_tag(tag))

# Number of break points
n = 17

# Reference points for regularization
A0 = np.linspace(-2, 2, n)
B0 = np.linspace(-2, 2, n)

# Save break points
db.bkpts = {"alpha": A0, "beta": B0}
# Get Mach number break points
db.get_bkpts("mach")

# Regularize
db.regularize_by_rbf(fmcols, ["alpha", "beta"], prefix="reg.", scol="mach")

# Print results
print("max error(reg.alpha) = %.2f" % np.max(db["reg.alpha"][:n*n:n] - A0))
print("max error(reg.beta)  = %.2f" % np.max(db["reg.beta"][:n] - B0))

