r"""
:mod:`cape.pyfun.options.databookopts`
======================================

DataBook options module specific to :mod:`cape.pyfun`.
"""

# Import base class
from ...cfdx.options import databookopts


# Alter one default for "FM"
class IterPointProbeDataBookOpts(databookopts.IterPointProbeDataBookOpts):
    # No attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Cols": ["p", "T"],
        "FloatCols": ["nOrders"],
    }


# Class for DataBook options
class DataBookOpts(databookopts.DataBookOpts):
    # Section map
    _sec_cls_optmap = {
        "PointProbe": IterPointProbeDataBookOpts,
    }

