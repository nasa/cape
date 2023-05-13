"""
:mod:`cape.pyover.options.Report`: pyOver Report Options
=========================================================

This module provides options for creating automated reports for pyOver results.
Although there are some subfigure types that are unique to pyOver, all of the
methods in the :class:`pyOver.options.Report.Report` class are inherited from
:mod:`cape` version.

See the :ref:`JSON Report <cape-json-Report>` section . for the list of
available figures and subfigures (along with other options).

:See Also:
    * :mod:`cape.cfdx.options.Report`
    * :mod:`cape.cfdx.report`
    * :mod:`cape.pyover.report`
"""


# Local imports
from ...cfdx.options import reportopts


# Class for L-infinity residual subfigure
class PlotLInfSubfigOpts(reportopts.ResidualSubfigOpts):
    # Defaults
    _rc = {
        "Residual": "Linf",
        "YLabel": r"$L_\infty$ residual",
    }


# Modify subfigure collection
class SubfigCollectionOpts(reportopts.SubfigCollectionOpts):
    # Modify class map
    _sec_cls_optmap = {
        "PlotLInf": PlotLInfSubfigOpts,
    }


# Class for Report settings
class ReportOpts(reportopts.ReportOpts):
    # Modify sections
    _sec_cls = {
        "Subfigures": SubfigCollectionOpts,
    }

