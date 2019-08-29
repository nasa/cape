#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import cape module
from cape.units import mks

# Basic Tests

# Inch to meter conversions
n = 1
u1 = "in"
u2 = "m"
v1 = 1.0
s1 = "inch"
# Status update
print("%02i (length): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.5f" % mks(s1))


# Horsepower and prefix
n = 2
u1 = "khp"
u2 = "W"
v1 = 1.0
s1 = "khp"
# Status update
print("%02i (imperial w/ mks prefix): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.5f" % mks(s1))


# Hectares
n = 3
u1 = "hectare"
u2 = "m^2"
v1 = 1.0
s1 = "hectare"
# Status update
print("%02i (area): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.3f" % mks(s1))


# Speeds
n = 4
u1 = "ft/mhr"
u2 = "m/s"
v1 = 1.0
s1 = "ft/mhr"
# Status update
print("%02i (slash): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.5f" % mks(s1))


# Unicode
n = 5
u1 = "um"
u2 = "m"
v1 = 1.0
s1 = u"μm"
# Status update
print("%02i (micro): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.2e" % mks(s1))


# Dual prefix
n = 6
u1 = "nGm"
u2 = "m"
v1 = 1.0
s1 = "nGm"
# Status update
print("%02i (double-prefix): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.2e" % mks(s1))


# Asterisk
n = 7
u1 = "ft*lbf"
u2 = "N*m"
v1 = 5.0
s1 = u1
# Status update
print("%02i (asterisk): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.4f" % (v1*mks(s1)))


# Space
n = 8
u1 = "ft*lbf"
u2 = "N*m"
v1 = 5.0
s1 = "ft lbf"
# Status update
print("%02i (space): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.4f" % (v1*mks(s1)))


# Carat
n = 9
u1 = "in^3"
u2 = "L"
v1 = 100.0
s1 = "in^3/L"
# Status update
print("%02i (exponent): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.4f" % (v1*mks(s1)))


# Included number
n = 10
u1 = "slug"
u2 = "kg"
v1 = 12.0
s1 = "12 slug"
# Status update
print("%02i (exponent): %.2f %s --> %s" % (n, v1, u1, u2))
# Display conversion
print("  %.4f" % (mks(s1)))

