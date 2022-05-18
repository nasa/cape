#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Local imports
from cape import atm


def test_h():
    # Set altitudes (in km)
    h0 = 0.0
    h1 = 2.0
    h2 = 26.0
    # Call the standard atmosphere
    s0 = atm.atm76(h0)
    s1 = atm.atm76(h1)
    s2 = atm.atm76(h2)
    # Check results
    assert abs(s0.p - 101325.0) <= 0.01
    assert abs(s1.p - 79498.14) <= 0.01
    assert abs(s2.p - 2187.96) <= 0.01
    assert abs(s0.rho - 1.225) <= 0.001
    assert abs(s1.a - 332.501) <= 0.001
    assert abs(s2.T - 222.5443) <= 1e-4


def test_sutherland_mks():
    # Get sea level conditions
    s0 = atm.atm76(0.0)
    # Calculate viscosity
    mu0 = atm.SutherlandMKS(s0.T)
    # Test value
    assert abs(mu0 - 1.7893e-5) <= 1e-8


def test_enthalpy():
    # Pick a reference specific enthalpy [J/kg]
    href = 1.2000e+06
    # Calculate temperature [K]
    Tref = atm.get_T(href)
    # Check results
    assert abs(Tref - 1184.5) <= 0.1
