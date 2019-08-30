#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
import pyOver


# Get control interface
cntl = pyOver.Overflow()

# Read the databook
cntl.ReadDataBook()

# Get the value
CN = cntl.DataBook['bullet_no_base']['CN'][0]

# STDOUT
print("CN = %0.3f" % CN)

