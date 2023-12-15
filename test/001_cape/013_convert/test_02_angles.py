# -*- coding: utf-8 -*-

# Local imports
import cape.convert


# Main tolerance
TOL = 1e-2
# Total angle of attack
ALPHA_T = 1.0
# Roll angle total
PHI = 45.0
# Angle of attack
ALPHA = (2**0.5)/2
# Sideslip angle
BETA = (2**0.5)/2


# Test (alpha_total, phi) <-> (alpha, beta)
def test_01_atp2ab():
    # Convert alpha total and phi to alpha/beta
    alpha, beta = cape.convert.AlphaTPhi2AlphaBeta(ALPHA_T, PHI)
    # Inverse conversion
    alpha_t, phi = cape.convert.AlphaBeta2AlphaTPhi(alpha, beta)
    # Test the directions are the same
    assert abs(ALPHA_T - alpha_t) <= TOL
    assert abs(PHI - phi) <= TOL


# Test (alpha_total, phi) <-> (u, v, w)
def test_02_atp2dc():
    # Convert alpha total and phi to directional cosines
    u, v, w = cape.convert.AlphaTPhi2DirectionCosines(ALPHA_T, PHI)
    # Inverse conversion
    alpha_t, phi = cape.convert.DirectionCosines2AlphaTPhi(u, v, w)
    # Test the directions are the same
    assert abs(ALPHA_T - alpha_t) <= TOL
    assert abs(PHI - phi) <= TOL


# Test (alpha, beta) <-> (u, v, w)
def test_03_ab2dc():
    # Convert alpha and beta to directional cosines
    u, v, w = cape.convert.AlphaBeta2DirectionCosines(ALPHA, BETA)
    # Inverse conversion
    alpha, beta = cape.convert.DirectionCosines2AlphaBeta(u, v, w)
    # Test the directions are the same
    assert abs(ALPHA - alpha) <= TOL
    assert abs(BETA - beta) <= TOL


# Test (alpha_total, phi) <-> (alpha_maneuver, phi_maneuver)
def test_04_atp2amp():
    # Convert alpha total and phi to alpha and phi maneuver
    alpha_m, phi_m = cape.convert.AlphaTPhi2AlphaMPhi(ALPHA_T, PHI)
    # Inverse conversion
    alpha_t, phi = cape.convert.AlphaMPhi2AlphaTPhi(alpha_m, phi_m)
    # Test the directions are the same
    assert abs(ALPHA_T - alpha_t) <= TOL
    assert abs(PHI - phi) <= TOL


# Test (alpha, beta) <-> (alpha_maneuver, phi_maneuver)
def test_05_ab2amp():
    # Convert alpha total and phi to alpha and phi maneuver
    alpha_m, phi_m = cape.convert.AlphaBeta2AlphaMPhi(ALPHA, BETA)
    # Convert to alpha/beta (inverse not implemented yet)
    alpha_t, phi = cape.convert.AlphaMPhi2AlphaTPhi(alpha_m, phi_m)
    # Test the directions are the same
    assert abs(ALPHA_T - alpha_t) <= TOL
    assert abs(PHI - phi) <= TOL
