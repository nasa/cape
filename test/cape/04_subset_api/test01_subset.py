#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
import cape

# Load interface
cntl = cape.Cntl()

# Display subsets
print(list(cntl.x.FilterString("b2")))
print(list(cntl.x.FilterWildcard("poweroff/m0*")))
print(list(cntl.x.FilterRegex("m.\.5.*b2")))
print(list(cntl.x.GetIndices(I=range(15,20), cons=["Mach%1==0.5"])))

