#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csvfile as csvfile

# Read CSV file
db = csvfile.CSVFile("wt-sample.csv")

# Case number
i = 2

# Get attributes
run = db["run"][i]
pt = db["pt"][i]
mach = db["mach"][i]
alph = db["alpha"][i]
beta = db["beta"][i]

# Create a string
print("run%03i.%02i_m%.3fa%.1fb%.1f" % (run, pt, mach, alph, beta))

