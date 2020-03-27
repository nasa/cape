#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-reg.mat")

# Test cols
cols = ["bullet.CN", "bullet.dCN"]

# Standard args
args = ["mach", "alpha", "beta"]
# Get Mach number break points
db.create_bkpts(args)

# Set evaluation
db.make_responses(cols, "linear", args)

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50

# Preprpocess
args1 = db._process_plot_args1("bullet.CN", 0.9, 1.9, -0.1)
args2 = db._process_plot_args1("bullet.CN", np.array([200, 238]))

# Display
print("Version 1:")
print("  col = '%s'" % args1[0])
print("  I   = %s" % args1[1])
print("  a   = %s" % list(args1[3]))
print("Version 2:")
print("  col = '%s'" % args2[0])
print("  I   = %s" % args2[1])
print("  a   = %s" % list(args2[3]))
