#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.rdb as rdb

# Read CSV file
db = rdb.DataKit("mab01.mat")

# Basic search
I, J = db.find(["mach", "alpha", "beta"], 0.9, 2.0, 0.0)

# Print indices
for i in I:
    print("Case %02i: CT=%.1f" % (i, db["CT"][i]))

