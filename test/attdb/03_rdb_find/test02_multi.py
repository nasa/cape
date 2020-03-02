#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import CSV module
import cape.attdb.rdb as rdb

# Read CSV file
db = rdb.DataKit("mab01.mat")

# Calculate break points
db.get_bkpts(["mach"], nmin=1)

# All Mach numbers
mach = db.bkpts["mach"]

# Set values
aoa = 2.0
aos = 0.0

# Basic search
I, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos)

# Title
print("Select alpha,beta, display all")
# Print indices
for i in I:
    # Get values
    m = db["mach"][i]
    a = db["alpha"][i]
    b = db["beta"][i]
    CT = db["CT"][i]
    # Create run name
    frun = "m%0.2fa%.1fb%.1f_CT%.1f" % (m, a, b, CT)
    # Display it
    print("  Case %02i: %s" % (i, frun))

# Search each condition once
I, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos, once=True)

# Title
print("Select alpha,beta, match once")
# Print indices
for i, j in zip(I, J):
    # Get values
    m = db["mach"][i]
    a = db["alpha"][i]
    b = db["beta"][i]
    CT = db["CT"][i]
    # Create run name
    frun = "m%0.2fa%.1fb%.1f_CT%.1f" % (m, a, b, CT)
    # Display it
    print("  Case %02i, match %02i: %s" % (i, j, frun))

# Search each condition once
Imap, J = db.find(["mach", "alpha", "beta"], mach, aoa, aos, mapped=True)

# Title
print("Select alpha,beta, map all matches")
# Print indices
for imap, j in zip(Imap, J):
    # Loop through map
    for i in imap:
        # Get values
        m = db["mach"][i]
        a = db["alpha"][i]
        b = db["beta"][i]
        CT = db["CT"][i]
        # Create run name
        frun = "m%0.2fa%.1fb%.1f_CT%.1f" % (m, a, b, CT)
        # Display it
        print("  Case %02i, match %02i: %s" % (i, j, frun))

