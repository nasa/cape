#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape.case

# Read conditions
x = cape.case.ReadConditions()

# Show conditions
print(x["mach"])
print(x["alpha"])

# Read conditions directly
beta = cape.case.ReadConditions('beta')
# Display
print(beta)
