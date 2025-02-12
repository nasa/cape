
# Standard library
import os

# Third-party
import cape.pltfile
import numpy as np


XCOLS = (
    "mach",
    "alpha",
    "beta",
    "q",
    "T",
    "config",
    "Label"
)

COEFFS = (
    "CA",
    "CY",
    "CN"
)


def f3dfunc(cntl, i):
    plt = cape.pltfile.Plt("arrow_tec_boundary_timestep200.plt")
    v = {}
    for i, zone in enumerate(plt.Zones):
        v[zone.split(" ")[-1] + ".cp_std"] = np.std(plt.q[i][:,-2])
    return v
