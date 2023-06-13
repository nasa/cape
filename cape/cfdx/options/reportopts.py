"""
:mod:`cape.cfdx.options.reportopts`: Automated report options
==============================================================

This module interfaces options for generating reports. Since many of the
report options are common to different solvers, much of the report
generation content is controlled here.

The function :func:`ReportOpts.SetSubfigDefaults` contains an extensive
set of default options for each subfigure type. However, the docstring
does not contain an outline or table of these, so interested users can
refer to the JSON documentation or read the source code for that
function.

"""

# Local imports
from ...optdict import OptionsDict, BOOL_TYPES, INT_TYPES, FLOAT_TYPES


# Options for a single report
class SingleReportOpts(OptionsDict):
    r"""Options to define title and list of figures for a single report

    :Versions:
        * 2023-04-20 ``@ddalle``: v1.0
    """
   # --- Class attributes ---
    # No instance attributes
    __slots__ = ()

    # Option list
    _optlist = (
        "Affiliation",
        "Archive",
        "Author",
        "ErrorFigures",
        "Figures",
        "Frontispiece",
        "Logo",
        "MinIter",
        "Parent",
        "Restriction",
        "ShowCaseNumber",
        "Subtitle",
        "Sweeps",
        "Title",
        "ZeroFigures",
    )

    # Types
    _opttypes = {
        "Affiliation": str,
        "Archive": BOOL_TYPES,
        "Author": str,
        "ErrorFigures": str,
        "Figures": str,
        "Frontispiece": str,
        "Logo": str,
        "MinIter": INT_TYPES,
        "Parent": str,
        "Restriction": str,
        "ShowCaseNumber": BOOL_TYPES,
        "Subtitle": str,
        "Sweeps": str,
        "Title": str,
        "ZeroFigures": str,
    }

    # List depth
    _optlistdepth = {
        "ErrorFigures": 1,
        "Figures": 1,
        "Sweeps": 1,
        "ZeroFigures": 1,
    }

    # Defaults
    _rc = {
        "Affiliation": "",
        "Archive": True,
        "Author": "",
        "MinIter": 0,
        "ShowCaseNumber": True,
        "Subtitle": "",
        "Title": "CAPE report",
    }

    # Descriptions
    _rst_descriptions = {
        "Affiliation": "organization for report authors",
        "Archive": "option to tar report folders after compilation",
        "Author": "automated report authors",
        "ErrorFigures": "list of figures for cases with ERROR status",
        "Figures": "list of figures in report",
        "Frontispiece": "image for repore title page",
        "Logo": "logo for footer of each report page",
        "MinIter": "minimum iteration for report to generate",
        "Parent": "name of report from which to inherit options",
        "Restriction": "distribution restriction label",
        "ShowCaseNumber": "option to show run matrix case index on each page",
        "Subtitle": "report subtitle",
        "Sweeps": "list of sweeps to include",
        "Title": "report title",
        "ZeroFigures": "list of figures for cases with 0 iterations",
    }


# Class for definition of a sweep
class SweepOpts(OptionsDict):
    r"""Options for a single report sweep definition

    This class controls the options that define a single report sweep.
    That includes the constraints that divide a run matrix into subsets
    and a list of figures to include for each individual subset. It
    controls each subsection of the *Report* > *Sweeps* CAPE options.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`SweepCollectionOpts`
    """
    # Attributes
    __slots__ = ()

    # Options
    _optlist = (
        "CarpetEqCons",
        "EqCons",
        "Figures",
        "IndexTol",
        "Indices",
        "GlobalCons",
        "MinCases",
        "RunMatrixOnly",
        "TolCons",
        "XCol",
        "YCol",
    )

    # Aliases
    _optmap = {
        "EqConstraints": "EqCons",
        "EqualityCons": "EqCons",
        "EqualityConstraints": "EqCons",
        "GlobalConstraints": "GlobalCons",
        "TolConstraints": "TolCons",
        "ToleranceConstraints": "TolCons",
        "XAxis": "XCol",
        "YAxis": "YCol",
        "cols": "EqCons",
        "cons": "GlobalCons",
        "figs": "Figures",
        "itol": "IndexTol",
        "mask": "Indices",
        "nmin": "MinCases",
        "tols": "TolCons",
        "xcol": "XCol",
        "xk": "XCol",
        "xkey": "XCol",
        "yk": "YCol",
        "ykey": "YCol",
    }

    # Types
    _opttypes = {
        "CarpetEqCons": str,
        "CarpetTolCons": dict,
        "EqCons": str,
        "Figures": str,
        "GlobalCons": str,
        "IndexTol": INT_TYPES,
        "Indices": INT_TYPES,
        "MinCases": INT_TYPES,
        "RunMatrixOnly": BOOL_TYPES,
        "TolCons": dict,
        "XCol": str,
        "YCol": str,
    }

    # List depth
    _optlistdepth = {
        "CarpetEqCons": 1,
        "EqCons": 1,
        "Figures": 1,
        "GlobalCons": 1,
        "Indices": 1,
    }

    # Defaults
    _rc = {
        "MinCases": 3,
        "RunMatrixOnly": False,
    }

    # Descriptions
    _rst_descriptions = {
        "CarpetEqCons": "run matrix keys that are constant on carpet subsweep",
        "CarpetTolCons": "tolerances for carpet subsweep by run matrix key",
        "EqCons": "list of run matrix keys that must be constant on a sweep",
        "Figures": "list of figures in sweep report",
        "GlobalCons": "list of global constraints for sweep",
        "IndexTol": "max delta of run matrix/databook index for single sweep",
        "Indices": "explicit list of run matrix/databook indices to include",
        "MinCases": "minimum number of data points in a sweep to include plot",
        "RunMatrixOnly": "option to restrict databook to current run matrix",
        "TolCons": "tolerances for run matrix keys to be in same sweep",
        "XCol": "run matrix key to use for *x*-axis of sweep plots",
        "YCol": "run matrix key to use for *y*-axis of sweep contour plots",
    }


# Collection of sweeps
class SweepCollectionOpts(OptionsDict):
    r"""Options for a collection of report sweep definitions

    This class is a :class:`dict` of :class:`SweepOpts` instances, and
    any there are no limitations on the keys. It controls the
    *Report* > *Sweeps* section of the full CAPE options interface.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`SweepOpts`
    """
    # Attributes
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": SweepOpts,
    }

    # Get option from a sweep
    def get_SweepOpt(self, sweep: str, opt: str, **kw):
        r"""Retrieve an option for a sweep

        :Call:
            >>> val = opts.get_SweepOpt(sweep, opt)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sweep*: :class:`str`
                Name of sweep
            *opt*: :class:`str`
                Name of option to retrieve
        :Outputs:
            *val*: :class:`object`
                Sweep option value
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
            * 2023-04-27 ``@ddalle``: v2.0; use ``optdict``
        """
        # Set parent key
        kw.setdefault("key", "Parent")
        # Recurse
        return self.get_subopt(sweep, opt, **kw)


# Class for definition of a figure
class FigureOpts(OptionsDict):
    r"""Options for a single report figure definition

    This class controls the options that define a single report figure.
    That includes several figure-specific options and a list of
    subfigures to include in the figure. It controls each subsection of
    the *Report* > *Figures* CAPE options.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`FigureCollectionOpts`
    """
    # Additional attibutes
    __slots__ = ()

    # Attribute list
    _optlist = (
        "Alignment",
        "Header",
        "Parent",
        "Subfigures",
    )

    # Aliases
    _optmap = {
        "Align": "Alignment",
        "align": "Alignment",
        "subfigs": "Subfigures",
    }

    # Types
    _opttypes = {
        "Alignment": str,
        "Header": str,
        "Parent": str,
        "Subfigures": str,
    }

    # Values
    _optvals = {
        "Alignment": {"left", "center", "right"},
    }

    # Defaults
    _rc = {
        "Alignment": "center",
        "Header": "",
    }

    # List depth
    _optlistdepth = {
        "Subfigures": 1,
    }

    # Descriptions
    _rst_descriptions = {
        "Alignment": "horizontal alignment for subfigs in a figure",
        "Header": "optional header for a figure",
        "Parent": "name of report from which to inherit options",
    }


# Class for list of figures
class FigureCollectionOpts(OptionsDict):
    r"""Options for a collection of figure definitions

    This class is a :class:`dict` of :class:`SubfigOpts` instances, and
    any there are no limitations on the keys. It controls the
    *Report* > *Figures* section of the full CAPE options interface.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`FigureOpts`
    """
    # Additional attibutes
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": FigureOpts,
    }

    # Get option from a figure
    def get_FigOpt(self, fig: str, opt: str, **kw):
        r"""Retrieve an option for a figure

        :Call:
            >>> val = opts.get_FigOpt(fig, opt)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
            *opt*: :class:`str`
                Name of option to retrieve
        :Outputs:
            *val*: :class:`object`
                Sweep option value
        :Versions:
            * 2023-04-27 ``@ddalle``: v1.0
        """
        # Set parent key
        kw.setdefault("key", "Parent")
        # Recurse
        return self.get_subopt(fig, opt, **kw)


# Class for subfigures
class SubfigOpts(OptionsDict):
    r"""Options for a single report subfigure definition

    This class controls the options that define a single report
    subfigure. This includes all the options that define what the
    subfigure should include. It controls each subsection of the
    *Report* > *Subfigures* CAPE options.

    There are many subclasses of :class:`SubfigOpts` that contain
    options appropriate for each subfigure type.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`FigureCollectionOpts`
    """
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "Alignment",
        "Caption",
        "Position",
        "Type",
        "Width",
    )

    # Aliases
    _optmap = {
        "Parent": "Type",
        "parent": "Type",
        "pos": "Position",
        "type": "Type",
        "width": "Width",
    }

    # Types
    _opttypes = {
        "Caption": str,
        "Position": str,
        "Type": str,
        "Width": FLOAT_TYPES,
    }

    # Permitted values
    _optvals = {
        "Position": ("t", "c", "b"),
    }

    # Defaults
    _rc = {
        "Alignment": "center",
        "Position": "b",
    }

    # Descriptions
    _rst_descriptions = {
        "Caption": "subfigure caption",
        "Position": "subfigure vertical alignment",
        "Type": "subfigure type or parent",
    }


# Class for table-type subfigures
class _TableSubfigOpts(SubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "Header",
    )

    # Types
    _opttypes = {
        "Header": str,
    }

    # Defaults
    _rc = {
        "Header": "",
    }

    # Descriptions
    _rst_descriptions = {
        "Header": "subfigure header",
    }


# Options for coefficient table
class CoeffTableSubfigOpts(_TableSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLN",
        "CLM",
        "Coefficients",
        "Components",
        "EpsFormat",
        "Iteration",
        "MuFormat",
        "SigmaFormat",
    )

    # List depth
    _optlistdepth = {
        "CA": 1,
        "CY": 1,
        "CN": 1,
        "CLL": 1,
        "CLN": 1,
        "CLM": 1,
        "Coefficients": 1,
        "Components": 1,
    }

    # Types
    _opttypes = {
        "CA": str,
        "CY": str,
        "CN": str,
        "CLL": str,
        "CLN": str,
        "CLM": str,
        "Coefficients": str,
        "Components": str,
        "EpsFormat": str,
        "Iteration": INT_TYPES,
        "MuFormat": str,
        "SigmaFormat": str,
    }

    # Default values
    _rc = {
        "CA": ["mu", "std"],
        "CY": ["mu", "std"],
        "CN": ["mu", "std"],
        "CLL": ["mu", "std"],
        "CLN": ["mu", "std"],
        "CLM": ["mu", "std"],
    }

    # Descriptions
    _rst_descriptions = {
        "Coefficients": "list of coefficients to detail in table",
        "Components": "list of components for which to report coefficients",
        "EpsFormat": "printf-style text format for sampling error",
        "Iteration": "specific iteration at which to sample results",
        "MuFormat": "printf-style text format for mean value",
        "SigmaFormat": "printf-sylte text format for standard deviation",
    }


# Options for conditions table
class ConditionsTableSubfigOpts(_TableSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "SkipVars",
        "SpecialVars",
    )

    # List depth
    _optlistdepth = {
        "SkipVars": 1,
        "SpecialVars": 1,
    }

    # Types
    _opttypes = {
        "SkipVars": str,
        "SpecialVars": str,
    }

    # Descriptions
    _rst_descriptions = {
        "SkipVars": "list of run matrix keys to leave out of table",
        "SpecialVars": "keys not in run matrix to attempt to calculate",
    }


# Options for sweep conditions tabel
class SweepConditionsSubfigOpts(ConditionsTableSubfigOpts):
    # Attributes
    __slots__ = ()


# Class to handle various PlotOptions dict
class _PlotOptsOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Aliases
    _optmap = {
        "c": "color",
        "ls": "linestyle",
        "lw": "linewidth",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "ms": "markersize",
    }

    # Defaults
    _rc = {
        "color": "k",
    }


# Options for iterative histories
class _MPLSubfigOpts(SubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "DPI",
        "FigureHeight",
        "FigureWidth",
        "Format",
        "NPlotFirst",
        "PlotOptions",
        "Restriction",
        "RestrictionLoc",
        "RestrictionOptions",
        "RestrictionXPosition",
        "RestrictionYPosition",
        "TickLabelOptions",
        "TickLabels",
        "Ticks",
        "XLabel",
        "XLabelOptions",
        "XLim",
        "XLimMax",
        "XMax",
        "XMin",
        "XTickLabelOptions",
        "XTickLabels",
        "XTicks",
        "YLabel",
        "YLabelOptions",
        "YLim",
        "YLimMax",
        "YMax",
        "Ymin",
        "YTickLabelOptions",
        "YTickLabels",
        "YTicks",
    )

    # Aliases
    _optmap = {
        "FigHeight": "FigureHeight",
        "FigWidth": "FigureWidth",
        "GridStyle": "GridPlotOptions",
        "LineOptions": "PlotOptions",
        "MinorGridStyle": "MinorGridPlotOptions",
        "RestrictionLocation": "RestrictionLoc",
        "RestrictionX": "RestrictionXPosition",
        "RestrictionY": "RestrictionYPosition",
        "dpi": "DPI",
        "nPlotFirst": "NPlotFirst",
        "nFirst": "NPlotFirst",
    }

    # Types
    _opttypes = {
        "DPI": INT_TYPES,
        "FigureHeight": FLOAT_TYPES,
        "FigureWidth": FLOAT_TYPES,
        "Format": str,
        "Grid": BOOL_TYPES,
        "GridPlotOptions": dict,
        "MinorGrid": BOOL_TYPES,
        "MinorGridPlotOptions": dict,
        "NPlotIters": INT_TYPES,
        "PlotOptions": dict,
        "Restriction": str,
        "RestrictionLoc": str,
        "RestrictionOptions": dict,
        "RestrictionXPosition": FLOAT_TYPES,
        "RestrictionYPosition": FLOAT_TYPES,
        "TickLabelOptions": dict,
        "TickLabels": BOOL_TYPES,
        "Ticks": BOOL_TYPES,
        "XLabel": str,
        "XLabelOptions": dict,
        "XMax": FLOAT_TYPES,
        "XMin": FLOAT_TYPES,
        "XTickLabelOptions": dict,
        "XTickLabels": (str,) + FLOAT_TYPES + BOOL_TYPES,
        "XTicks": FLOAT_TYPES + BOOL_TYPES,
        "YLabel": str,
        "YLabelOptions": dict,
        "YMax": FLOAT_TYPES,
        "YMin": FLOAT_TYPES,
        "YTickLabelOptions": dict,
        "YTickLabels": (str,) + FLOAT_TYPES + BOOL_TYPES,
        "YTicks": FLOAT_TYPES + BOOL_TYPES,
    }

    # Permissible values
    _optvals = {
        "Format": ("pdf", "svg", "png", "jpg", "jpeg"),
        "RestrictionLoc": (
            "bottom",
            "bottom left",
            "bottom right",
            "left",
            "lower right",
            "lower left",
            "right",
            "top",
            "top left",
            "top right",
            "upper left",
            "upper right",
        ),
    }

    # Defaults
    _rc = {
        "DPI": 150,
        "Format": "pdf",
        "FigureWidth": 6,
        "FigureHeight": 4.5,
        "GridPlotOptions": {},
        "MinorGridPlotOptions": {},
        "Restriction": "",
        "RestrictionLoc": "top",
        "RestrictionOptions": {},
    }

    # Descriptions
    _rst_descriptions = {
        "DPI": "dots per inch if saving as rasterized image",
        "FigureHeight": "height of subfigure graphics in inches",
        "FigureWidth": "width of subfigure graphics in inches",
        "Format": "image file format",
        "NPlotFirst": "iteration at which to start figure",
        "PlotOptions": "options for main line(s) of plot",
        "TickLabelOptions": "common options for ticks of both axes",
        "TickLabels": "common value(s) for ticks of both axes",
        "Restriction": "data restriction to place on figure",
        "RestrictionLoc": "location for subfigure restriction text",
        "RestrictionOptions": "additional opts to ``text()`` for restriction",
        "RestrictionXPosition": "explicit x-coord of restriction",
        "RestrictionYPosition": "explicit y-coord of restriction",
        "XLabel": "manual label for x-axis",
        "XLabelOptions": "text options for x-axis label",
        "XLim": "explicit min and max limits for x-axis",
        "XLimMax": "outer limits for min and max x-axis limits",
        "XMax": "explicit upper limit for x-axis limits",
        "XMin": "explicit lower limit for x-axis limits",
        "XTickLabelOptions": "text options for x-axis tick labels",
        "XTickLabels": "option to turn off x-axis tick labels or set values",
        "XTicks": "option to turn off x-axis ticks or set values",
        "YLabel": "manual label for y-axis",
        "YLabelOptions": "text options for y-axis label",
        "YLim": "explicit min and max limits for y-axis",
        "YLimMax": "outer limits for min and max y-axis limits",
        "YMax": "explicit upper limit for y-axis limits",
        "YMin": "explicit lower limit for y-axis limits",
        "YTickLabelOptions": "text options for y-axis tick labels",
        "YTickLabels": "option to turn off x-axis tick labels or set values",
        "YTicks": "option to turn off y-axis ticks or set values",
    }


# Options for generic Matplotlib figs
class _IterSubfigOpts(_MPLSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "NPlotFirst",
        "NPlotIters",
        "NPlotLast",
    )

    # Aliases
    _optmap = {
        "nPlotFirst": "NPlotFirst",
        "nFirst": "NPlotFirst",
        "nPlotIters": "NPlotIters",
        "nPlotLast": "NPlotLast",
    }

    # Types
    _opttypes = {
        "NPlotFirst": INT_TYPES,
        "NPlotIters": INT_TYPES,
        "NPlotLast": INT_TYPES,
    }

    # Defaults
    _rc = {
        "NPlotFirst": 1,
    }

    # Descriptions
    _rst_descriptions = {
        "NPlotFirst": "iteration at which to start figure",
    }


# Plot options for residual (e.g. L2) plots
class ResidPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "linewidth": 1.5,
        "linestyle": "-",
        "color": "k",
    }


# Plot options for residual (e.g. L2) plots
class ResidPlot0Opts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "linewidth": 1.2,
        "linestyle": "-",
        "color": "b",
    }


# Options for residual plots
class ResidualSubfigOpts(_IterSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "PlotOptions0",
        "Residual",
    )

    # Aliases
    _optmap = {
        "LineOptions0": "PlotOptions0",
    }

    # Types
    _opttypes = {
        "PlotOptions0": dict,
        "Residual": str,
        "PlotOptions": ResidPlotOpts,
        "PlotOptions0": ResidPlot0Opts,
    }
    # Defaults
    _rc = {
        "Residual": "L2",
    }

    # Subclasses
    _sec_cls = {

    }

    # Descriptions
    _rst_descriptions = {
        "PlotOptions0": "plot options for initial residual",
        "Residual": "name of residual field or type to plot",
    }


# Options for residual plots
class PlotL1SubfigOpts(ResidualSubfigOpts):
    # Attributes
    __slots__ = ()
    # Defaults
    _rc = {
        "Residual": "L1",
    }


# Plot options for coefficient plots
class PlotCoeffPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "color": ["k", "g", "c", "m", "b", "r"],
    }


# Sigma Plot options for coefficient plots
class PlotCoeffSigmaPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "facecolor": "b",
        "alpha": 0.35,
        "ls": "none",
    }


# Options for plotting a coefficient, either iter or sweep
class _PlotCoeffSubfigOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "Coefficient",
        "Component",
        "KSigma",
        "PlotOptions",
        "SigmaPlotOptions",
    )

    # Aliases
    _optmap = {
        "LineOptions": "PlotOptions",
        "Sigma": "KSigma",
        "StDevOptions": "SigmaPlotOptions",
        "StandardDeviation": "KSigma",
        "col": "Coefficient",
        "ksig": "KSigma",
        "nSigma": "KSigma",
        "sig": "NSigma",
        "sigma": "KSigma",
    }

    # Types
    _opttypes = {
        "Coefficient": str,
        "Component": str,
        "KSigma": FLOAT_TYPES,
        "PlotOptions" : PlotCoeffPlotOpts,
        "SigmaPlotOptions": PlotCoeffSigmaPlotOpts,
    }

    # Defaults
    _rc = {
        "Component": "entire",
        "KSigma": 0.0,
    }

    # Descriptions
    _rst_descriptions = {
        "Coefficient": "column(s) to plot iterative history of",
        "Component": "component(s) for which to plot *Coefficient*",
        "KSigma": "multiple of sigma to plot above and below mean",
        "PlotOptions": "line plot options",
        "SigmaPlotOptions": "plot options for standard deviation box",
    }


# Delta Plot options for coefficient iteration plots
class PlotCoeffIterDeltaPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "color": None,
    }


# Epsilon Plot options for coefficient iteration plots
class PlotCoeffIterEpsilonPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "facecolor": "g",
         "alpha": 0.4,
         "ls": "none",
    }


# Mu Plot options for coefficient iteration plots
class PlotCoeffIterMuPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "ls": "none",
    }


# Options for other iterative value plots
class PlotCoeffIterSubfigOpts(_IterSubfigOpts, _PlotCoeffSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "CaptionComponent",
        "Delta",
        "DeltaFormat",
        "DeltaPlotOptions",
        "EpsilonFormat",
        "EpsilonPlotOptions",
        "KEpsilon",
        "MuFormat",
        "MuPlotOptions",
        "NAverage",
        "ShowDelta",
        "ShowEpsilon",
        "ShowMu",
        "ShowSigma",
        "SigmaFormat",
    )

    # Aliases
    _optmap = {
        "ErrPltOptions": "EpsilonPlotOptions",
        "ErrorFormat": "EpsilonFormat",
        "DeltaOptions": "DeltaPlotOptions",
        "IterativeError": "KEpsilon",
        "LineOptions": "PlotOptions",
        "MeanOptions": "MuPlotOptions",
        "NAvg": "NAverage",
        "StDevOptions": "SigmaPlotOptions",
        "col": "Coefficient",
        "nAverage": "NAverage",
        "nAvg": "NAverage",
        "nEpsilon": "KEpsilon",

    }

    # Types
    _opttypes = {
        "CaptionComponent": str,
        "Delta": FLOAT_TYPES,
        "DeltaFormat": str,
        "DeltaPlotOptions": PlotCoeffIterDeltaPlotOpts,
        "EpsilonFormat": str,
        "EpsilonPlotOptions": PlotCoeffIterEpsilonPlotOpts,
        "KEpsilon": FLOAT_TYPES,
        "MuFormat": str,
        "MuPlotOptions": PlotCoeffIterMuPlotOpts,
        "NAverage": INT_TYPES,
        "ShowDelta": BOOL_TYPES,
        "ShowEpsilon": BOOL_TYPES,
        "ShowMu": BOOL_TYPES,
        "ShowSigma": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "Delta": 0.0,
        "DeltaFormat": "%.4f",
        "EpsilonFormat": "%.4f",
        "KEpsilon": 0.0,
        "MuFormat": "%.4f",
        "ShowMu": [True, False],
        "ShowSigma": [True, False],
        "ShowDelta": [True, False],
        "ShowEpsilon": False,
        "SigmaFormat": "%.4f",
    }

    # Descriptions
    _rst_descriptions = {
        "CaptionComponent": "explicit text for component portion of caption",
        "Delta": "specified interval(s) to plot above and below mean",
        "DeltaFormat": "printf-style flag for *ShowDelta value",
        "DeltaPlotOptions": "plot options for fixed-width above and below mu",
        "EpsilonFormat": "printf-style flag for *ShowEpsilon* value",
        "EpsilonOptions": "plot options for sampling error box",
        "KEpsilon": "multiple of iterative error to plot",
        "MuFormat": "printf-style flag for *ShowMu* value",
        "MuPlotOptions": "plot options for horizontal line showing mean",
        "ShowDelta": "option to print value of *Delta*",
        "ShowEpsilon": "option to print value of iterative sampling error",
        "ShowMu": "option to print value of mean over window",
        "ShowSigma": "option to print value of standard deviation",
        "SigmaFormat": "printf-style flag for *ShowSigma* value",
    }



# MinMax Plot options for coefficient sweep plots
class PlotCoeffSweepMinMaxPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {}

# Target Plot options for coefficient sweep plots
class PlotCoeffSweepTargetPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {}

# Options for sweep value plots
class PlotCoeffSweepSubfigOpts(_MPLSubfigOpts, _PlotCoeffSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "MinMax",
        "MinMaxOptions",
        "Target",
        "TargetOptions",
    )

    # Types
    _opttypes = {
        "MinMax": BOOL_TYPES,
        "MinMaxOptions": PlotCoeffSweepMinMaxPlotOpts,
        "Target": str,
        "TargetOptions": PlotCoeffSweepTargetPlotOpts,
    }

    # Defaults
    _rc = {
        "MinMax": False,
    }

    # Descriptions
    _rst_descriptions = {
        "MinMax": "option to plot min/max of value over iterative window",
        "MinMaxOptions": "plot options for *MinMax* plot",
        "Target": "name of target databook to co-plot",
        "TargetOptions": "plot options for optional target",
    }


# Seam Curve Plot options for lineload plots
class PlotLineLoadSeamPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {}



# Options for line load plots
class PlotLineLoadSubfigOpts(_MPLSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "AdjustBottom",
        "AdjustLeft",
        "AdjustRight",
        "AdjustTop",
        "AutoUpdate",
        "Coefficient",
        "Component",
        "Orientation",
        "SeamCurve",
        "SeamOptions",
        "SeamLocation",
        "SubplotMargin",
        "XPad",
        "YPad",
    )

    # Aliases
    _optmap = {
        "SeamCurves": "SeamCurve",
        "SeamLocations": "SeamLocation",
        "SeamCurveOptions": "SeamOptions",
        "Targets": "Target",
    }

    # Types
    _opttypes = {
        "AdjustBottom": FLOAT_TYPES,
        "AdjustLeft": FLOAT_TYPES,
        "AdjustRight": FLOAT_TYPES,
        "AdjustTop": FLOAT_TYPES,
        "AutoUpdate": BOOL_TYPES,
        "Coefficient": str,
        "Component": str,
        "Orientation": str,
        "SeamCurve": str,
        "SeamOptions": PlotLineLoadSeamPlotOpts,
        "SeamLocation": str,
        "SubplotMargin": FLOAT_TYPES,
        "XPad": FLOAT_TYPES,
        "YPad": FLOAT_TYPES,
    }

    # Permissible values
    _optvals = {
        "Orientation": ("horizontal", "vertical"),
        "SeamCurve": ("smy", "smz"),
        "SeamLocation": ("bottom", "left", "right", "top"),
    }

    # Defaults
    _rc = {
        "AdjustBottom": 0.1,
        "AdjustLeft": 0.12,
        "AdjustRight": 0.97,
        "AdjustTop": 0.97,
        "AutoUpdate": True,
        "Orientation": "vertical",
        "SubplotMargin": 0.015,
        "XPad": 0.03,
        "YPad": 0.03,
    }

    # Descriptions
    _rst_descriptions = {
        "AdjustBottom": "margin from axes to bottom of figure",
        "AdjustLeft": "margin from axes to left of figure",
        "AdjustRight": "margin from axes to right of figure",
        "AdjustTop": "margin from axes to top of figure",
        "AutoUpdate": "option to create line loads if not in databook",
        "Coefficient": "coefficient to plot",
        "Component": "config component tp plot",
        "Orientation": "orientation of vehicle in line load plot",
        "SeamCurve": "name of seam curve, if any, to show w/ line loads",
        "SeamLocation": "location for optional seam curve plot",
        "SeamOptions": "plot options for optional seam curve",
        "SubplotMargin": "margin between line load and seam curve subplots",
        "XPad": "additional padding from data to xmin and xmax w/i axes",
        "YPad": "additional padding from data to ymin and ymax w/i axes",
    }


# Seam Curve Plot options for lineload plots
class ContourCoeffPlotOpts(_PlotOptsOpts):
    # Attributes
    __slots__ = ()

    # Defaults
    _rc = {
        "color": "k",
        "lw": 0,
        "marker": "o",
        "markersize": 4,
    }

# Options for contour plots
class PlotContourCoeffSubfigOpts(_MPLSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "AxisEqual",
        "ColorBar",
        "ContourColorMap",
        "ContourOptions",
        "ContourType",
        "LineType",
        "XCol",
        "YCol",
    )

    # Aliases
    _optmap = {
        "ColorMap": "ContourColorMap",
        "PlotType": "LineType",
        "XAxis": "XCol",
        "YAxis": "YCol",
        "xcol": "XCol",
        "xk": "XCol",
        "ycol": "YCol",
        "yk": "YCol",
    }

    # Types
    _opttypes = {
        "AxisEqual": BOOL_TYPES,
        "ColorBar": BOOL_TYPES,
        "ContourColorMap": str,
        "ContourOptions": dict,
        "ContourType": str,
        "LineType": "plot",
        "PlotOptions": ContourCoeffPlotOpts,
        "XCol": str,
        "YCol": str,
    }

    # Permissible values
    _optvals = {
        "ContourType": ("tricontour", "tricontourf", "tripcolor"),
        "LineType": ("plot", "triplot"),
    }

    # Defaults
    _rc = {
        "AxisEqual": True,
        "ColorBar": True,
        "ContourColorMap": "jet",
        "ContourType": "tricontourf",
        "LineType": "plot",
    }

    # Descriptions
    _rst_descriptions = {
        "AxisEqual": "option to scale x and y axes with common scale",
        "ColorBar": "option to turn on color bar (scale)",
        "ContourColorMap": "name of color map to use w/ contour plots",
        "ContourOptions": "options passed to contour plot function",
        "ContourType": "contour plotting function/type to use",
        "LineType": "plot function to use to mark data points",
        "XCol": "run matrix key to use for *x*-axis",
        "YCol": "run matrix key to use for *y*-axis",
    }


# Tecplot subfigure
class TecplotSubfigOpts(SubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "ColorMaps",
        "ContourLevels",
        "FieldMap",
        "FigWidth",
        "Keys",
        "Layout",
        "VarSet",
    )

    # Types
    _opttypes = {
        "ColorMaps": dict,
        "ContourLevels": dict,
        "FieldMap": INT_TYPES,
        "FigWidth": INT_TYPES,
        "Keys": dict,
        "Layout": str,
        "VarSet": dict,
    }

    # List depth
    _optlistdepth = {
        "ColorMaps": 1,
        "ContourLevels": 1,
        "FieldMap": 1,
    }

    # Defaults
    _rc = {
        "ColorMaps": [],
        "FigWidth": 1024,
        "Width": 0.5,
        "VarSet": {},
    }

    # Descriptions
    _rst_descriptions = {
        "ColorMaps": "customized Tecplot colormap",
        "ContourLevels": "customized settings for Tecplot contour levels",
        "FieldMap": "list of zone numbers for Tecplot layout group boundaries",
        "FigWidth": "width of output image in pixels",
        "Keys": "dict of Tecplot layout statements to customize",
        "Layout": "template Tecplot layout file",
        "VarSet": "variables and their values to define in Tecplot layout",
    }


# Paraview subfigure
class ParaviewSubfigOpts(SubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "Command",
        "Format",
        "ImageFile",
        "Layout",
    )

    # Types
    _opttypes = {
        "Command": str,
        "Format": str,
        "ImageFile": str,
        "Layout": str,
    }

    # Defaults
    _rc = {
        "Command": "pvpython",
        "Format": "png",
        "ImageFile": "export.png",
        "Layout": "layout.py",
        "Width": 0.5,
    }

    # Descriptions
    _rst_descriptions = {
        "Command": "name of Python/Paraview executable to call",
        "Format": "image file format",
        "ImageFile": "name of image file created by *Layout*",
        "Layout": "name of Python file to execute with Paraview",
    }


# Existing-image subfigure
class ImageSubfigOpts(SubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "ImageFile",
    )

    # Types
    _opttypes = {
        "ImageFile": str,
    }

    # Defaults
    _rc = {
        "ImageFile": "export.png",
        "Width": 0.5,
    }

    # Descriptions
    _rst_descriptions = {
        "ImageFile": "name of image file to copy from case folder",
    }


# Class for subfigure collections
class SubfigCollectionOpts(OptionsDict):
    r"""Options for a collection of subfigure definitions

    This class is a :class:`dict` of :class:`SubfigOpts` instances, and
    any there are no limitations on the keys. It controls the
    *Report* > *Subfigures* section of the full CAPE options interface.

    See also:
        * :class:`cape.optdict.OptionsDict`
        * :class:`ReportOpts`
        * :class:`SubfigOpts`
    """
    # No attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Section classes
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": SubfigOpts,
        "CoeffTable": CoeffTableSubfigOpts,
        "Conditions": ConditionsTableSubfigOpts,
        "ConditionsTable": ConditionsTableSubfigOpts,
        "ContourCoeff": PlotContourCoeffSubfigOpts,
        "FMTable": CoeffTableSubfigOpts,
        "Image": ImageSubfigOpts,
        "Paraview": ParaviewSubfigOpts,
        "PlotCoeff": PlotCoeffIterSubfigOpts,
        "PlotCoeffIter": PlotCoeffIterSubfigOpts,
        "PlotCoeffSweep": PlotCoeffSweepSubfigOpts,
        "PlotContour": PlotContourCoeffSubfigOpts,
        "PlotContourSweep": PlotContourCoeffSubfigOpts,
        "PlotL1": PlotL1SubfigOpts,
        "PlotL2": ResidualSubfigOpts,
        "PlotLineLoad": PlotLineLoadSubfigOpts,
        "PlotResid": ResidualSubfigOpts,
        "Summary": CoeffTableSubfigOpts,
        "SweepCases": SweepConditionsSubfigOpts,
        "SweepCoeff": PlotCoeffSweepSubfigOpts,
        "SweepConditions": SweepConditionsSubfigOpts,
        "Tecplot": TecplotSubfigOpts,
    }

    # Get option from a subfigure
    def get_SubfigOpt(self, sfig: str, opt: str, j=None, **kw):
        r"""Retrieve an option for a subfigure

        :Call:
            >>> val = opts.get_SubfigOpt(sfig, opt, j=None, **kw)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
            *j*: {``None``} | :class:`int`
                Phase index
        :Outputs:
            *val*: :class:`object`
                Sweep option value
        :Versions:
            * 2023-04-27 ``@ddalle``: v1.0
        """
        # Set parent key
        kw.setdefault("key", "Type")
        # Recurse
        return self.get_subopt(sfig, opt, j=j, **kw)

    # Get base type of a figure
    def get_SubfigBaseType(self, sfig: str):
        r"""Get root type for an individual subfigure

        :Call:
            >>> t = opts.get_SubfigBaseType(sfig)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *t*: :class:`str`
                Subfigure parent type
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # Name of option ("Type")
        key = self.__class__._sec_cls_opt
        # Tunnel down to lowest level of "Type"
        return self.get_subkey_base(sfig, key)

    # Return all non-default options for a subfigure
    def get_SubfigCascade(self, sfig: str):
        r"""Return full set of optsion from subfig and its parents

        :Call:
            >>> sfigopts = opts.get_SubfigCasecasde(sfig)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *sfigopts*: :class:`dict`
                Options for subfigure *sfig*
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2023-05-10 ``@ddalle``: v2.0; ``optdict`` rewrites
        """
        # Check if present
        if sfig not in self:
            raise KeyError("No subfigure called '%s" % sfig)
        # Get a copy of subfigure options
        sfigopts = dict(self[sfig])
        # Get the type, which may be a parent subfigure
        parent = sfigopts.get("Type")
        # Check if that type is also defined
        if parent not in self:
            # No cascade; probably found the "BaseType"
            return sfigopts
        # Get the options from that subfigure; recurse
        parentopts = self.get_SubfigCascade(parent)
        # Get new type
        parent2 = parentopts.get("Type")
        # Overwrite "Type"
        if parent2 is not None:
            sfigopts["Type"] = parent2
        # Apply template options but do not overwrite
        for key, val in parentopts.items():
            sfigopts.setdefault(key, val)
        # Output
        return sfigopts


# Class for complete *Report* section
class ReportOpts(OptionsDict):
    r"""Dictionary-based interface for *Reports* section

    :Versions:
        * 2016-30-02 ``@ddalle``: v1.0
        * 2023-04-28 ``@ddalle``: v2.0; converted to ``OptionsDict``
    """
   # --- Class attributes ---
    # Attribute list
    __slots__ = (
        "sfig",
    )

    # Option list
    _optlist = {
        "Figures",
        "Reports",
        "Subfigures",
        "Sweeps",
    }

    # Aliases
    _optmap = {}

    # Option types
    _opttypes = {
        "_default_": SingleReportOpts,
        "Figures": dict,
        "Subfigures": dict,
        "Sweeps": dict,
        "Reports": str,
    }

    # List depth
    _optlistdepth = {
        "Reports": 1,
    }

    # Defaults
    _rc = {
        "Archive": True,
    }

    # Option to add allowed options
    _xoptkey = "Reports"

    # Subsection classes
    _sec_cls = {
        "Figures": FigureCollectionOpts,
        "Subfigures": SubfigCollectionOpts,
        "Sweeps": SweepCollectionOpts,
    }

    # Descriptions
    _rst_descriptions = {
        "Figures": "collection of figure definitions",
        "Reports": "list of reports",
        "Subfigures": "collection of subfigure definitions",
        "Sweeps": "collection of sweep definitions",
    }

   # --- Dunder ---
    # Initialization method
    def __init__(self, *args, **kw):
        r"""Initialization method

        :Call:
            >>> opts = Report(**kw)
        :Inputs:
            *kw*: :class:`dict` | :class:`odict`
                Dictionary that is converted to this class
        :Outputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v2.0; simple OptionsDict method
        """
        # Initialize
        OptionsDict.__init__(self, *args, **kw)
        # Store self subfigure tag
        self.sfig = None

   # --- Lists ---
    # List of reports
    def get_ReportList(self, j=None, **kw):
        r"""Get list of reports available to create

        :Call:
            >>> reps = opts.get_ReportList()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *reps*: :class:`list`\ [:class:`str`]
                List of reports by name
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v2.0; simple OptionsDict method
        """
        # Set default None -> []
        vdef = kw.pop("vdef", [])
        # Output
        return self.get_opt("Reports", j=j, vdef=vdef, **kw)

    # List of sweeps
    def get_SweepList(self) -> list:
        r"""Get list of sweeps for a report

        :Call:
            >>> fswps = opts.get_SweepList()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *figs*: :class:`list`\ [:class:`str`]
                List of figures by name
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v2.0; Updates for OptionsDict
        """
        # Get sweep definitions
        sweepopts = self.get("Sweeps", {})
        # Output the keys as a list
        return [sweep for sweep in sweepopts]

    # List of figures (case)
    def get_FigList(self) -> list:
        r"""Get list of figures for a report

        :Call:
            >>> figs = opts.get_FigList()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *figs*: :class:`list`\ [:class:`str`]
                List of figures by name
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v2.0; Updates for OptionsDict
        """
        # Get figures options
        figopts = self.get("Figures", {})
        # Output the keys as a list
        return [fig for fig in figopts]

    # List of available subfigures
    def get_SubfigList(self) -> list:
        r"""Get list of available subfigures for a report

        :Call:
            >>> figs = opts.get_SubfigList()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *sfigs*: :class:`list`\ [:class:`str`]
                List of subfigures by name
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2023-04-20 ``@ddalle``: v2.0; Updates for OptionsDict
        """
        # Get figures dictionary
        sfigopts = self.get('Subfigures', {})
        # Output the keys as a list
        return [sfig for sfig in sfigopts]

   # --- Report definitions ---
    # Get report option
    def get_ReportOpt(self, report: str, opt: str, **kw):
        r"""Get named option for a specific report

        :Call:
            >>> val = opts.get_ReportOpt(report, opt, **kw)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *report*: :class:`str`
                Name of report
            *opt*: :class:`str`
                Name of option
        :Outputs:
            *val*: :class:`object`
                Value of *opt*
        :Versions:
            * 2023-04-27 ``@ddalle``: v1.0
        """
        # Use *Parent*
        kw.setdefault("key", "Parent")
        # Options
        return self.get_subopt(report, opt, **kw)

    # Get report list of figures for cases marked FAIL
    def get_ReportErrorFigures(self, report: str):
        r"""Get list of figures for cases with ERROR status

        :Call:
            >>> figs = opts.get_ReportErrorFigList(report)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *report*: :class:`str`
                Name of report
        :Outputs:
            *figs*: :class:`list`\ [:class:`str`]
                List of figures in the report
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # Get suboption
        figs = self.get_subopt(report, "ErrorFigures", key="Parent")
        # If empty, fall back to main figure list
        if figs is None:
            figs = self.get_subopt(report, "Figures", key="Parent")
        # Output
        return figs

   # --- Figures ---
   # --- Sweeps ---
   # --- Subfigures ---


# Promote subsections
ReportOpts.promote_sections()
