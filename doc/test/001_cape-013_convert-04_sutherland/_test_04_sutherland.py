# -*- coding: utf-8 -*-

# Local imports
import cape.convert
from cape.units import mks

TOL = 1e-6
# Room temp in Rankine
RT_R = 528.0
# Room temp in Kelvin
RT_K = 293.333


# Test Sutherland's law
def test_01_sutherlands():
    # Room temp viscosity in imperial
    mu_fps = cape.convert.SutherlandFPS(528.0)
    # Room temp viscosity in metric
    mu_mks = cape.convert.SutherlandMKS(293.333)
    # Check these are the same if converted to metric
    assert abs(mu_mks - mu_fps*mks("slug/ft*s")) <= TOL
