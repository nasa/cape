#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
from cape.units import mks

# Basic Tests

# Test 1 simple units
print("01: Simple Conversions")
print("inch to meter")
# print
m = mks("inch")
print(m)

# Test 2
print("horsepower to Watt")
hp = mks("khp")
print(hp)

# Test 3
print("hectare to meter")
hm = mks("hacre")
print(hm)
    
print("02: More complicated units")
# Test 4
print("ft/s to m/s")
# print
ms = mks("ft/s")
print(ms)

# Test 5 
print("psi to N/(m*m)")
# print
psi = mks("psi")
print(psi)

print("03: Conversions with numbers")
# Test 6
print("slinch to kg")
# technically not a slinch since mks only supports slug
kg = mks("12 slug")
print(kg)   

# Test 7 
print("518 Rankine to Kelvin")
k = 518*mks("R")
print (k)

# Test 8
print("50 revs per hour to rad/s")
rads = mks("50 rev/hr")
print(rads)
