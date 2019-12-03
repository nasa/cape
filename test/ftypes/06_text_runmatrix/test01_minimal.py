#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.textdata as td

# Read as generic text with special first-column flag
db = td.TextDataFile("runmatrix.csv",
    FirstColBoolMap={"PASS": "p", "ERROR": "e"})

# Case number
i = 6

# Get attributes
for col in db.cols:
    print("%8s: %s" % (col, db[col][i]))

