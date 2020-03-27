#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.dbfm as dbfm


# Read CSV file
db = dbfm.DBFM("bullet-fm-mab.mat")

# Standard args
args = ["mach"]
# Mapped cols
qcols = ["q", "T"]
# Get Mach number break points
db.create_bkpts(args)
# Get break points for dynamic pressure and tempreature
db.create_bkpts_map(qcols, "mach")

# Set evaluation
db.make_responses(qcols, "linear", args, eval_kwargs={"bkpt": True})

# Pick some conditions
mach = 0.90

# Conditions
print("mach: %.2f" % mach)
# Loop throuch columns
for col in qcols:
    # Print it
    print("%s: %.2f" % (col, db(col, mach)))

