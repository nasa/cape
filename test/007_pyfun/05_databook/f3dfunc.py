
# Third-party
import cape.pltfile
import numpy as np


def f3dfunc(cntl, i):
    plt = cape.pltfile.Plt("arrow_tec_boundary_timestep200.plt")
    v = {}
    for i, zone in enumerate(plt.Zones):
        v[zone.split(" ")[-1] + ".cp_std"] = np.std(plt.q[i][:, -2])
    return v
