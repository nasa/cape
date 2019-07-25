#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.atm

# Get sea level conditions
s0 = cape.atm.atm76(0.0)
# Calculate viscosity
mu0 = cape.atm.SutherlandMKS(s0.T)

# Print results
print("%0.2e" % mu0)
