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
db.set_output_xargs("bullet.dCN", ["bullet.x"])

# Pick some conditions
mach = 0.90
alph = 1.50
beta = 0.50

# Preprpocess
args1 = db._prep_args_plot1("bullet.CN", 0.9, 1.9, -0.1)
args2 = db._prep_args_plot1("bullet.CN", np.array([200, 238]))
args3 = db._prep_args_plot1("bullet.dCN", np.array([240, 251]))

# Unpack
mach1, aoa1, beta1 = args1[3]
mach2, aoa2, beta2 = args2[3]
mach3, aoa3, beta3 = args3[3]

# Display
print("Version 1: scalar by scalar args")
print("  col  = '%s'" % args1[0])
print("  I    = %s" % args1[1])
print("  mach = %5.2f" % mach1)
print("  aoa  = %5.2f" % aoa1)
print("  beta = %5.2f" % beta1)
print("Version 2: scalar by array indices")
print("  col  = '%s'" % args2[0])
print("  I    = %s" % ", ".join(["%i" % x for x in args2[1]]))
print("  mach = %s" % ", ".join(["%5.2f" % x for x in mach2]))
print("  aoa  = %s" % ", ".join(["%5.2f" % x for x in aoa2]))
print("  beta = %s" % ", ".join(["%5.2f" % x for x in beta2]))
print("Version 3:")
print("  col = '%s'" % args3[0])
print("  I    = %s" % ", ".join(["%i" % x for x in args3[1]]))
print("  mach = %s" % ", ".join(["%5.2f" % x for x in mach3]))
print("  aoa  = %s" % ", ".join(["%5.2f" % x for x in aoa3]))
print("  beta = %s" % ", ".join(["%5.2f" % x for x in beta3]))

