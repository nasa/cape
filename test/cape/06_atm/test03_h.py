#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.atm

# Pick a reference specific enthalpy [J/kg]
href = 1.2000e+06

# Calculate temperature [K]
Tref = cape.atm.get_T(href)

# Print results
print("%0.2e" % Tref)
