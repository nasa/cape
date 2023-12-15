# -*- coding: utf-8 -*-

# Local imports
import cape.convert

# Absolute tolerance
TOL = 1e-6
# Supersonic Mach
MACH1 = 1.5
# Subsonic Mach
MACH2 = 0.610432


# Test Critical Mach
def test_01_arearatios():
    # Get critical area ratio
    ar = cape.convert.CriticalAreaRatio(MACH1)
    # Get supersonic exit mach for this ratio for mach 1 input
    mach_super = cape.convert.ExitMachFromAreaRatio(ar, 1.0)
    # Get subsonic exit mach for this ratio for mach 1 input
    mach_sub = cape.convert.ExitMachFromAreaRatio(ar, 1.0, subsonic=True)
    # Check machs are within tolerance
    assert abs(MACH1 - mach_super) <= TOL
    # Check machs are within tolerance
    assert abs(MACH2 - mach_sub) <= TOL

