r"""
:mod:`cape.pyfun.options.databookopts`
======================================

DataBook options module specific to :mod:`cape.pyfun`.
"""

# Import base class
from ...cfdx.options import databookopts


# Alter one default for "FM"
class DBFMOpts(databookopts.DBFMOpts):
    # No attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "Cols": ["CA", "CY", "CN", "CLL", "CLM", "CLN"],
        "FloatCols": ["nOrders"],
    }


# Class for DataBook options
class DataBookOpts(databookopts.DataBookOpts):
    # Section map
    _sec_cls_optmap = {
        "FM": DBFMOpts,
        "IterPoint": databookopts.DBIterPointOpts,
        "LineLoad": databookopts.DBLineLoadOpts,
        "PyFunc": databookopts.DBPyFuncOpts,
        "TriqFM": databookopts.DBTriqFMOpts,
        "TriqPoint": databookopts.DBTriqPointOpts,
    }

