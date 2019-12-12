#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.xls as xls

# Read CSV file
db = xls.XLSFile("aero_arrow_no_base.xlsx",
    Types={
        "config": "str",
        "alpha": "float16",
        "mach": "float32",
        "nStats": "int"
    })

# Print data types
for col in db.cols:
    # Get array
    V = db[col]
    # Check type
    clsname = V.__class__.__name__
    # Data type
    dtype = V[0].__class__.__name__
    # Status message
    print("%8s: %s (%s)" % (col, dtype, clsname))

