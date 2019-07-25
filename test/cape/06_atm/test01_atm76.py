#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.atm

# Set altitudes (in km)
h0 = 0.0
h1 = 2.0
h2 = 26.0

# Call the standard atmosphere
s0 = cape.atm.atm76(h0)
s1 = cape.atm.atm76(h1)
s2 = cape.atm.atm76(h2)

# Print results
print("%0.2f" % s0.p)
print("%0.2f" % s1.p)
print("%0.2f" % s2.p)
print("%0.2f" % s0.rho)
print("%0.2f" % s1.a)
print("%0.2f" % s2.T)
