#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.ftypes.csvfile as csvfile

# Read CSV file
db = csvfile.CSVFile("aeroenv.csv",
    DefaultType="float32",
    Definitions={"beta": {"Type": "float16"}})

# Print data types
for col in db.cols:
    print("%-5s: %s" % (col, db[col].dtype.name))

