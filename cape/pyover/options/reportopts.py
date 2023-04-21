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


# Class for Report settings
class ReportOpts(reportopts.ReportOpts):
    # Modify defaults
    def ModSubfigDefaults(self):
        r"""Modify subfigure defaults for OVERFLOW

        :Call:
            >>> opts.SetSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
        :Effects:
            *opts.defns*: :class:`dict`
                Some subfigure default options sets are reset
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        # Plot L2 residual
        self.defs['PlotL2'] = {
            "Header": "",
            "Residual": "L2",
            "Component": None,
            "Position": "b",
            "Alignment": "center",
            "YLabel": "L2 residual",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
        # Plot L2 residual
        self.defs['PlotLInf'] = {
            "Header": "",
            "Residual": "L2",
            "Component": None,
            "Position": "b",
            "Alignment": "center",
            "YLabel": "L2 residual",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
        # Plot general residual
        self.defs["PlotResid"] = {
            "Residual": "L2",
            "Component": None,
            "YLabel": "Residual",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
        # Plot general turbulence residual
        self.defs["PlotTurbResid"] = {
            "Residual": "L2",
            "Component": None,
            "YLabel": "Residual",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
        # Plot general species residual
        self.defs["PlotSpeciesResid"] = {
            "Residual": "L2",
            "Component": None,
            "YLabel": "Residual",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150
        }
# class Report


