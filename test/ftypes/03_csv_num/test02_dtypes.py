#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csvfile as csvfile

# Read CSV file
db = csvfile.CSVFile("wt-sample.csv", DefaultType="int32")

# Print data types
for col in db.cols:
    print("%-5s: %s" % (col, db[col].dtype.name))

