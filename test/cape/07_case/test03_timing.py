#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library modules
import datetime

# Import cape module
import cape.cfdx.case as case

# Example file names
fstrt = "cape_start.dat"
ftime = "cape_time.dat"

# Create initial time
tic = datetime.datetime.now()

# Read settings
rc = case.ReadCaseJSON()

# Write a flag for starting a program
case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")

# Read it
nProc, t0 = case.ReadStartTimeProg(fstrt)

# Calculate delta time
dt = tic - t0

# Test output
print("%i cores, %.4f seconds" % (nProc, dt.seconds))

# Write output file
case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")

