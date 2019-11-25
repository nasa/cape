#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csv as csv

# Read CSV file
db = csv.CSVFile("wt-sample.csv", DefaultType="int32")

# Print data types
for col in db.cols:
    print("%-5s: %s" % (col, db[col].dtype.name))

