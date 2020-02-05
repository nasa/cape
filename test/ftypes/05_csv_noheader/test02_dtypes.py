#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csvfile as csvfile

# Read CSV file
db = csvfile.CSVFile("runmatrix.csv",
    cols=["mach", "alpha", "beta", "config", "Label", "user"],
    DefaultType="float32",
    Types={
        "config": "str",
        "mach": "float16"
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

