#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library modules
import sys

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
db.regularize_by_griddata(
    fmcols, ["alpha", "beta"], prefix="reg.", scol="mach")

# Get new column names
regcols = [col for col in db.cols if col.startswith("reg.")]
# Number of cols per row
mcol = 3
# Number of reg cols
ncol = len(regcols)
# Number of rows
nrow = (ncol + mcol - 1) // mcol
# Print results
print("regularized cols:")
# Loop through rows of printed columns
for i in range(nrow):
    # Indent
    sys.stdout.write("  ")
    # Loop through columns
    for j in range(mcol):
        # Index
        icol = i*mcol + j
        # Check for uneven row, reached end of list
        if icol >= ncol:
            break
        # Get col
        col = regcols[icol]
        # Display it
        sys.stdout.write("  %-14s: %i" % (col, db[col].size))
    # New line
    sys.stdout.write("\n")
# Flush
sys.stdout.flush()
    
