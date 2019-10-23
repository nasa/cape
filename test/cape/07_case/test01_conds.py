#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.cfdx.case

# Read conditions
x = cape.cfdx.case.ReadConditions()

# Show conditions
print(x["mach"])
print(x["alpha"])

# Read conditions directly
beta = cape.cfdx.case.ReadConditions('beta')
# Display
print(beta)
