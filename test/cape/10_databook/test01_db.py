#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import numpy
import numpy as np

# Import cape modules
import cape
import cape.dataBook

# Test DataBook CLass
# Read settings
cntl = cape.Cntl()

# Read data book
DB = cape.dataBook.DataBook(cntl.x, cntl.opts)

# Display the data book
print(DB)
# List the components
for comp in DB.Components:
    print(comp)

# Extract a component
DBc = DB["fin1"]
# Display that
print(DBc)

# Match the trajectory to the actual data
DBc.UpdateTrajectory()
# Filter cases at alpha=2
I = DBc.x.Filter(["alpha==2"])
# Show those cases
for frun in DBc.x.GetFullFolderNames(I):
    print(frun)

# Test CaseFM Class
# Create a force & moment history
FM = cape.dataBook.CaseFM("fin")
# Some iterations
n = 500
# Create some iterations
FM.i = np.arange(n)

# Seed the random number generator
np.random.seed(450)
# Create some random numbers
FM.CN = 1.4 + 0.3*np.random.randn(n)

# Save properties
FM.cols = ["i", "CN"]
FM.coeffs = ["CN"]

# Display it
print(FM)

# Calculate statistics
S = FM.GetStatsN(100)

# Show values
print("Mean value: %(CN).4f" % S)
print("Min value: %(CN_min).4f" % S)
print("Max value: %(CN_max).4f" % S)
print("Standard deviation: %(CN_std).4f" % S)
print("Sampling error: %(CN_err).4f" % S)