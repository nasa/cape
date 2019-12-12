#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.xls as xls

# Read CSV file
db = xls.XLSFile("aero_arrow_no_base.xls")

# Case number
i = 6

# Get attributes
for col in db.cols:
    # Get value
    v = db[col][i]
    # Check type
    if isinstance(v, float):
        # Just use a few decimals
        print("%8s: %.2f" % (col, v))
    else:
        # Print default
        print("%8s: %s" % (col, v))
