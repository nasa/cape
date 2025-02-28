
# Third-party
import cape.pyover.casecntl
import numpy as np


def oflfunc(cntl, i):
    qt = cape.pyover.casecntl.checkqt("q.save")
    v = {}
    v["2x.its"] = qt*2
    return v
