#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.ftypes.xlsfile as xlsfile

# Read CSV file
db = xlsfile.XLSFile("header_categories.xlsx")

# Get attributes
for col in db.cols:
    # Get value
    v = db[col]
    # Check type
    if isinstance(v, np.ndarray):
        # Convert shape to string
        shape = "x".join([str(s) for s in v.shape])
        # Get data type
        dtype = v.dtype
        # Print name, type, and size
        print("    %-21s: array (shape=%s, dtype=%s)" % (col, shape, dtype))
    else:
        # Length
        n = len(v)
        # Type of first entry
        if n > 0:
            dtype = v[0].__class__.__name__
        else:
            dtype = "str"
        # Print type and size
        print("    %-21s: list (len=%i, type=%s)" % (col, n, dtype))

