#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
import cape.attdb.rdb as rdb

# Read DataKit from TSV file
db = rdb.DataKit(tsv="CN-dense.tsv")

# Print columns
for col in db.cols:
    # Get the data type
    print("%s: %s" % (col, db.get_col_prop(col, "Type")))

