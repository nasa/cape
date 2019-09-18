#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CAPE modules
import cape.pyfun


# Get control interface
cntl = cape.pyfun.Cntl()

# Read the databook
cntl.ReadDataBook()

# Get the value
CA = cntl.DataBook['bullet_no_base']['CA'][0]

# STDOUT
print("CA = %0.3f" % CA)

