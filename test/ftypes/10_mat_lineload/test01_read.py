#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import sys

# Import MAT module
import cape.attdb.ftypes.matfile as matfile

# Read MAT file
db = matfile.MATFile("bullet.mat")

# Get columns
if sys.version_info.major == 2:
    # Can't control column order ...
    cols = sorted(db.cols)
else:
    # Column order under our control
    cols = db.cols
# Loop through colums
for col in cols:
    # Display it and its shape
    print("%-10s: %s" % (col, db[col].shape))

