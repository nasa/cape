#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import MAT module
import cape.attdb.ftypes.matfile as matfile

# Read MAT file
db = matfile.MATFile("wt-sample.mat", DefaultType="int32")

# Print data types
for col in db.cols:
    print("%-5s: %s" % (col, db[col].dtype.name))

