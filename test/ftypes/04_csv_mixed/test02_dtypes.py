#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csv as csv

# Read CSV file
db = csv.CSVFile("runmatrix.csv",
    DefaultType="float32",
    Definitions={
        "config": {"Type": "str"},
        "Label": {"Type": "str"},
        "user": {"Type": "str"}
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

