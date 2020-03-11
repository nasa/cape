#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Third-party modules
import numpy as np

# Import CSV module
import cape.attdb.rdb as rdb


# Read CSV file
db = rdb.DataKit("CN-alpha-beta.mat")

# Number of break points
n = 17

# Reference points for regularization
A0 = np.linspace(-2, 2, n)
B0 = np.linspace(-2, 2, n)

# Save break points
db.bkpts = {"alpha": A0, "beta": B0}

# Regularize
db.regularize_by_rbf("CN", ["alpha", "beta"], prefix="reg")

# Print results
print("max error(regalpha) = %.2f" % np.max(db["regalpha"][::n] - A0))
print("max error(regbeta)  = %.2f" % np.max(db["regbeta"][:n] - B0))
print("monotonic(regCN): %s" % (np.min(np.diff(db["regCN"][::n]) > 0)))

