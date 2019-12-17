#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.rdbnull as rdbnull

# Read CSV file
db = rdbnull.DBResponseNull("aero_arrow_no_base.csv")

# Case number
i = 13

# Get attributes
mach = db["mach"][i]
alph = db["alpha"][i]
CA = db["CA"][i]

# Create a string
print("Case %i: m%.2fa%.2f CA=%.3f" % (i, mach, alph, CA))

# Write MAT file
db.write_mat("aero_arrow_no_base.mat")

# Reread
db1 = rdbnull.DBResponseNull("aero_arrow_no_base.mat")

# Get attributes
mach = db1["mach"][i]
alph = db1["alpha"][i]
CA = db1["CA"][i]

# Create a string
print("Case %i: m%.2fa%.2f CA=%.3f" % (i, mach, alph, CA))

