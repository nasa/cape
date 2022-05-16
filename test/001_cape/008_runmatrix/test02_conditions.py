#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Third party modules
import numpy as np

# Import cape module
import cape.runmatrix


# Test 01: convert *aoap*, *phip* to *a*, *b*
def test01_ab():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["aoap", "phip"],
        aoap=np.array([0.0, 4.0, 4.0, 4.0, 4.0]),
        phip=np.array([0.0, 0.0, 45.0, 90.0, 235.0]))
    # Status update
    print("01: aoap, phip --> a, b")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        aoap = x.GetAlphaTotal(i)
        phip = x.GetPhi(i)
        a = x.GetAlpha(i)
        b = x.GetBeta(i)
        # Display
        print("%i: aoap=%3.1f, phip=%5.1f -> a=%7.4f, b=%7.4f"
            % (i, aoap, phip, a, b))


# Test 02: convert *a*, *b* to *aoap*, *phip*
def test02_ab():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["alpha", "beta"],
        alpha=np.array([0.0, 4.0, 4.0, -4.0]),
        beta=np.array([0.0, 0.0, 4.0, -2.0]))
    # Status update
    print("02: a, b --> aoap, phip")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        aoap = x.GetAlphaTotal(i)
        phip = x.GetPhi(i)
        a = x.GetAlpha(i)
        b = x.GetBeta(i)
        # Display
        print("%i: a=%4.1f, b=%4.1f -> aoap=%4.2f, phip=%6.2f"
            % (i, a, b, aoap, phip))


# Test 03: convert *a*, *b* to *aoav*, *phiv*
def test03_ab():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["alpha", "beta"],
        alpha=np.array([0.0, 4.0, 4.0, -4.0]),
        beta=np.array([0.0, 0.0, 4.0, -2.0]))
    # Status update
    print("03: a, b --> aoav, phiv")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        aoav = x.GetAlphaManeuver(i)
        phiv = x.GetPhiManeuver(i)
        a = x.GetAlpha(i)
        b = x.GetBeta(i)
        # Display
        print("%i: a=%4.1f, b=%4.1f -> aoav=%5.2f, phiv=%6.2f"
            % (i, a, b, aoav, phiv))


# Test 04: convert *mach*, *q* to *p*
def test04_mq():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["mach", "alpha", "beta", "q"],
        mach=2.0,
        alpha=0.0,
        beta=0.0,
        q=np.array([100.0, 250.0, 300.0]))
    # Status update
    print("04: mach, q --> p, p0")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        m = x.GetMach(i)
        q = x.GetDynamicPressure(i)
        # Conversions
        p = x.GetPressure(i)
        p0 = x.GetTotalPressure(i)
        # Display
        print("%i: mach=%.2f, q=%5.1f psf -> p=%6.2f, p0=%6.2f"
            % (i, m, q, p, p0))


# Test 05: convert *mach*, *q*, *T* to *Re*
def test05_reynolds():
    # Create run matrix
    x = cape.runmatrix.RunMatrix(
        Keys=["mach", "alpha", "beta", "q", "T"],
        mach=2.0,
        alpha=0.0,
        beta=0.0,
        q=np.array([100.0, 250.0, 300.0]),
        T=450.0)
    # Status update
    print("04: mach, q --> p, p0")
    # Display folder names
    for i in range(x.nCase):
        # Get conditions
        m = x.GetMach(i)
        q = x.GetDynamicPressure(i)
        T = x.GetTemperature(i)
        # Conversions
        Re = x.GetReynoldsNumber(i)
        # Display
        print("%i: mach=%.2f, q=%5.1f psf  T=%5.1f R -> Rey=%.1f/in"
            % (i, m, q, T, Re))


if __name__ == "__main__":
    test01_ab()
    test02_ab()
    test03_ab()
    test04_mq()
    test05_reynolds()

