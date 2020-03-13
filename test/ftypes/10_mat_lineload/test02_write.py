#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import MAT module
import cape.attdb.ftypes.matfile as matfile

# Read MAT file
db = matfile.MATFile("bullet.mat")

# Rename a column
db["MACH"] = db.pop("mach")
db.cols.remove("mach")
db.cols.append("MACH")

# Write it
db.write_mat("bullet1.mat")

# Reread
db = matfile.MATFile("bullet1.mat")

# Loop through colums
for col in db.cols:
    # Display it and its shape
    print("%-10s: %s" % (col, db[col].shape))

