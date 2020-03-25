#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.rdb as rdb


# Read CSV file
db = rdb.DataKit("bullet-mab.mat")

# Get key fields
x = db["bullet.x"]
dCY = db["bullet.dCY"]
dCN = db["bullet.dCN"]
# MRP
xmrp = np.max(x)
# xhat for moments
xhat = xmrp - x
# Initialize moments
dCLL = np.zeros_like(dCN)
dCLM = np.zeros_like(dCN)
dCLN = np.zeros_like(dCY)

# Dimensions
_, n = dCY.shape

# Create moments for each condition
for i in range(n):
    # Create pitching and yawing moments
    dCLM[:,i] = xhat * dCN[:,i]
    dCLN[:,i] = xhat * dCY[:,i]

# Save moments
db.save_col("bullet.dCLL", dCLL)
db.save_col("bullet.dCLM", dCLM)
db.save_col("bullet.dCLN", dCLN)
# Save definitions
db.make_defn("bullet.dCLL", dCLL)
db.make_defn("bullet.dCLM", dCLM)
db.make_defn("bullet.dCLN", dCLN)

# Integrate all fields
for col in ["CA", "CY", "CN", "CLL", "CLM", "CLN"]:
    # Integrate
    db.create_integral("bullet.d%s" % col, "bullet.x")

# Save it
db.write_mat("bullet-FM-LL.mat")

# Number of columns
ncol = len(db.cols)
# Cols per row
mcol = 4
# Print the column names
print("cols:")
# Loop through rows of columns
for i in range((ncol + mcol - 1) // mcol):
    # Indent
    sys.stdout.write("   ")
    # Names
    for j in range(mcol):
        sys.stdout.write(" %-11s" % db.cols[i*mcol + j])
    # New line
    sys.stdout.write("\n")
    sys.stdout.flush()

# Pick a case
i = 9
# Print some results
print("values:")
for col in ["mach", "alpha", "beta", "bullet.CN", "bullet.CLM"]:
    # Print condition
    print("%11s: %.2f" % (col, db[col][i]))
