r"""
:mod:`cape.pyfun.options.databookopts`
======================================

DataBook options module specific to :mod:`cape.pyfun`.
"""

# Import base class
from ...cfdx.options import databookopts


# Alter one default for "FM"
class FMDataBookOpts(databookopts.FMDataBookOpts):
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
        "FM": FMDataBookOpts,
        "IterPoint": databookopts.DBIterPointOpts,
        "LineLoad": databookopts.LineLoadDataBookOpts,
        "PyFunc": databookopts.PyFuncDataBookOpts,
        "TimeSeries": databookopts.DBTimeSeriesOpts,
        "TriqFM": databookopts.TriqFMDataBookOpts,
        "TriqPoint": databookopts.DBTriqPointOpts,
    }

