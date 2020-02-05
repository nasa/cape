#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csvfile as csvfile

# Read CSV file
db = csvfile.CSVFile("aeroenv.csv")

# Case number
i = 13

# Get attributes
mach = db["mach"][i]
alph = db["alpha"][i]
beta = db["beta"][i]

# Create a string
print("m%.2fa%.2fb%.2f" % (mach, alph, beta))

