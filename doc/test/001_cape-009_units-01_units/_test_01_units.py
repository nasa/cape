# -*- coding: utf-8 -*-

# Local imports
from cape.units import mks


# Main tolerance
TOL = 1e-8


# Inch
def test_01_inch():
    # Inch to meter conversions
    s1 = "inch"
    # Check
    assert abs(mks(s1) - 0.0254) <= TOL


# Horsepower and prefix
def test_02_hp():
    # Target and actual
    v1 = 745699.87238
    v2 = mks("khp")
    assert abs(1 - v1/v2) <= TOL


# Hectares
def test_03_hectare():
    v1 = 10000.0
    v2 = mks("hectare")
    assert abs(v1 - v2) <= TOL


# Speeds
def test_04_slash():
    # This is feet per millihour
    v1 = 0.0846666667
    v2 = mks("ft/mhr")
    assert abs(1 - v1/v2) <= TOL


# Unicode
def test_05_unicode():
    v1 = 1e-6
    v2 = mks(u"μm")
    assert abs(1 - v1/v2) <= TOL


# Dual prefix
def test_06_doubleprefix():
    v1 = 1.0
    v2 = mks("nGm")
    assert abs(1 - v1/v2) <= TOL


# Asterisk
def test_07_asterix():
    v1 = 6.77908975
    v2 = 5.0 * mks("ft*lbf")
    assert abs(1 - v1/v2) <= TOL


# Space
def test_08_space():
    v1 = 6.77908975
    v2 = 5.0 * mks("ft lbf")
    assert abs(1 - v1/v2) <= TOL


# Carat
def test_09_carat():
    v1 = 0.016387064
    v2 = mks("in^3/L")
    assert abs(1 - v1/v2) <= TOL


# Included number
def test_10_number():
    v1 = 175.126835433
    v2 = mks("12 slug")
    assert abs(1 - v1/v2) <= TOL

