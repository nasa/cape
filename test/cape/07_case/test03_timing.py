#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library modules
import datetime

# Import cape module
import cape.case

# Example file names
fstrt = "cape_start.dat"
ftime = "cape_time.dat"

# Create initial time
tic = datetime.datetime.now()

# Read settings
rc = cape.case.ReadCaseJSON()

# Write a flag for starting a program
cape.case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")

# Read it
nProc, t0 = cape.case.ReadStartTimeProg(fstrt)

# Calculate delta time
dt = tic - t0

# Test output
print("%i cores, %.4f seconds" % (nProc, dt.seconds))

# Write output file
cape.case.WriteUserTimeProg(tic, rc, 0, ftime, "cape")

