
# Third-party
import cape.trifile
import numpy as np


def c3dfunc(cntl, i):
    triq = cape.trifile.Triq("Components.i.triq")
    v = {}
    v["cp.std"] = np.std(triq.q[:,0])
    return v
