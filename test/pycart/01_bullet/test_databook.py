#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
import cape.pycart


# Get control interface
cntl = pyCart.Cntl()

# Read the databook
cntl.ReadDataBook()

# Get the value
CA = cntl.DataBook['bullet_no_base']['CA'][0]

# STDOUT
print("CA = %0.3f" % CA)

