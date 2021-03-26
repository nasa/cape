#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard libraries
import sys

# Import third-party libraries
import numpy as np

# Import cape module
import cape.tri
import cape.config

# Output tri file
TRIFILE = "arrow-tri10k.lr4.tri"

# JSON file
JSONCONFIGFILE = "arrow.json"

# Read triangulation output from test01
tri = cape.tri.Tri(fname=TRIFILE)

# Check unique CompIDs
tricids = np.unique(tri.CompID)
print("CompIDs from TRI file")
# Print unique CompIDs from tri
print(tricids)

# Read JSON configuation
cfgj = cape.config.ConfigJSON(fname="arrow.json")
# Check unique CompIDs
jcids = np.unique(cfgj.IDs)
print("CompIDs from JSON file")
# Print unique CompIDs from JSON
print(jcids)
