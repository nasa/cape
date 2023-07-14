# -*- coding: utf-8 -*-

# Local imports
import cape.convert
from cape.units import mks

# Absolute tolerance
TOL = 1e-7
# Relative tolerance
RELTOL = 1e-4
# Mach
MACH = 0.2
# Reynolds Number
REY = 4555203.333
# Temperature imperial [R]
T_IMP = 528.0
# Temperature metric [K]
T_MET = 293.33333
# Pressure imperial [psf]
P_IMP = 2116.21662
# Pressure metric [Pa]
P_MET = 101325.0


# Test reynolds
def test_01_reynolds():
    # Get Reynolds per foot
    Re_f = cape.convert.ReynoldsPerFoot(P_IMP, T_IMP, MACH)
    # Get Reynolds per meter
    Re_m = cape.convert.ReynoldsPerMeter(P_MET, T_MET, MACH)
    # Get pressure imperial from Reynolds
    P_imp = cape.convert.PressureFPSFromRe(Re_f, T_IMP, MACH)
    # Get pressure metric from Reynolds
    P_met = cape.convert.PressureMKSFromRe(Re_m, T_MET, MACH)
    # Test relative error between Reynolds numbers
    assert abs(Re_m - Re_f * mks("1/ft")) / Re_m <= RELTOL
    # Test difference between pressures
    assert abs(P_IMP - P_imp) <= TOL
    assert abs(P_MET - P_met) <= TOL
