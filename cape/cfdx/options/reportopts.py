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
    _optmap = {
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
        "Coefficients",
        "Components",
        "EpsFormat",
        "Iteration",
        "MuFormat",
        "SigmaFormat",
    )

    # List depth
    _optlistdepth = {
        "Coefficients": 1,
        "Components": 1,
    }

    # Types
    _opttypes = {
        "Coefficients": str,
        "Components": str,
        "EpsFormat": str,
        "Iteration": INT_TYPES,
        "MuFormat": str,
        "SigmaFormat": str,
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


# Options for iterative histories
class _IterSubfigOpts(SubfigOpts):
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
        "XLabel",
        "YLabel",
    )

    # Aliases
    _optmap = {
        "FigHeight": "FigureHeight",
        "FigWidth": "FigureWidth",
        "LineOptions": "PlotOptions",
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
        "NPlotFirst": INT_TYPES,
        "PlotOptions": dict,
        "XLabel": str,
        "YLabel": str,
    }

    # Permissible values
    _optvals = {
        "Format": ("pdf", "svg", "png", "jpg", "jpeg"),
    }

    # Defaults
    _rc = {
        "DPI": 150,
        "Format": "pdf",
        "FigureWidth": 6,
        "FigureHeight": 4.5,
    }

    # Descriptions
    _rst_descriptions = {
        "DPI": "dots per inch if saving as rasterized image",
        "FigureHeight": "height of subfigure graphics in inches",
        "FigureWidth": "width of subfigure graphics in inches",
        "Format": "image file format",
        "NPlotFirst": "iteration at which to start figure",
        "PlotOptions": "options for main line(s) of plot",
        "XLabel": "manual label for x-axis",
        "YLabel": "manual label for y-axis",
    }


# Options for residual plots
class ResidualSubfigOpts(_IterSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "Residual",
    )

    # Types
    _opttypes = {
        "Residual": str,
    }

    # Defaults
    _rc = {
        "Residual": "L2",
    }

    # Descriptions
    _rst_descriptions = {
        "Residual": "name of residual field or type to plot",
    }


# Options for other iterative value plots
class IterSubfigOpts(_IterSubfigOpts):
    # Attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "CaptionComponent",
        "Coefficient",
        "Component",
        "Delta",
        "DeltaFormat",
        "DeltaPlotOptions",
        "EpsilonFormat",
        "EpsilonPlotOptions",
        "KEpsilon",
        "KSigma",
        "MuFormat",
        "MuPlotOptions",
        "NAverage",
        "ShowDelta",
        "ShowEpsilon",
        "ShowMu",
        "ShowSigma",
        "SigmaFormat",
        "SigmaPlotOptions",
    )

    # Aliases
    _optmap = {
        "ErrPltOptions": "EpsilonPlotOptions",
        "ErrorFormat": "EpsilonFormat",
        "IterativeError": "KEpsilon",
        "LineOptions": "PlotOptions",
        "MeanOptions": "MuPlotOptions",
        "NAvg": "NAverage",
        "Sigma": "KSigma",
        "StDevOptions": "SigmaPlotOptions",
        "StandardDeviation": "NSigma",
        "col": "Coefficient",
        "ksig": "KSigma",
        "nAverage": "NAverage",
        "nAvg": "NAverage",
        "nEpsilon": "KEpsilon",
        "nSigma": "KSigma",
        "sig": "NSigma",
        "sigma": "NSigma",
    }

    # Types
    _opttypes = {
        "CaptionComponent": str,
        "Coefficient": str,
        "Component": str,
        "Delta": FLOAT_TYPES,
        "DeltaFormat": str,
        "DeltaPlotOptions": dict,
        "EpsilonFormat": str,
        "EpsilonPlotOptions": dict,
        "KEpsilon": FLOAT_TYPES,
        "KSigma": FLOAT_TYPES,
        "MuFormat": dict,
        "MuPlotOptions": dict,
        "NAverage": INT_TYPES,
        "ShowDelta": BOOL_TYPES,
        "ShowEpsilon": BOOL_TYPES,
        "ShowMu": BOOL_TYPES,
        "ShowSigma": BOOL_TYPES,
        "SigmaPlotOptions": dict,
    }

    # Defaults
    _rc = {
        "Component": "entire",
        "Delta": 0.0,
        "DeltaFormat": "%.4f",
        "DeltaPlotOptions": {"color": None},
        "EpsilonFormat": "%.4f",
        "EpsilonPlotOptions": {"facecolor": "g", "alpha": 0.4, "ls": "none"},
        "KEpsilon": 0.0,
        "KSigma": 0.0,
        "MuFormat": "%.4f",
        "MuPlotOptions": {"ls": None},
        "PlotOptions": {"color": ["k", "g", "c", "m", "b", "r"]},
        "ShowMu": [True, False],
        "ShowSigma": [True, False],
        "ShowDelta": [True, False],
        "ShowEpsilon": False,
        "SigmaFormat": "%.4f",
        "SigmaPlotOptions": {"facecolor": "b", "alpha": 0.35, "ls": "none"},
        "Grid": None,
        "GridStyle": {},
        "MinorGrid": None,
        "MinorGridStyle": {}
    }

    # Descriptions
    _rst_descriptions = {
        "CaptionComponent": "explicit text for component portion of caption",
        "Coefficient": "column(s) to plot iterative history of",
        "Component": "component(s) for which to plot *Coefficient*",
        "Delta": "specified interval(s) to plot above and below mean",
        "DeltaFormat": "printf-style flag for *ShowDelta value",
        "DeltaPlotOptions": "plot options for fixed-width above and below mu",
        "EpsilonFormat": "printf-style flag for *ShowEpsilon* value",
        "EpsilonOptions": "plot options for sampling error box",
        "KEpsilon": "multiple of iterative error to plot",
        "KSigma": "multiple of sigma to plot above and below mean",
        "MuFormat": "printf-style flag for *ShowMu* value",
        "MuPlotOptions": "plot options for horizontal line showing mean",
        "ShowDelta": "option to print value of *Delta*",
        "ShowEpsilon": "option to print value of iterative sampling error",
        "ShowMu": "option to print value of mean over window",
        "ShowSigma": "option to print value of standard deviation",
        "SigmaFormat": "printf-style flag for *ShowSigma* value",
        "SigmaPlotOptions": "plot options for standard deviation box",
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

    # Section classes
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": SubfigOpts,
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
        return self.get_subopt(sfig, opt, **kw)

    # Get base type of a figure
    def get_SubfigBaseType(self, sfig):
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
        "defs",
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
        # Initialize subfigure defaults
        self.SetSubfigDefaults()
        self.ModSubfigDefaults()
        # Store self subfigure tag
        self.sfig = None
        self.defs = None

   # --- Defaults ---
    # Subfigure defaults
    def SetSubfigDefaults(self):
        r"""Set subfigure default options

        :Call:
            >>> opts.SetSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Effects:
            *opts.defns*: :class:`dict`
                Default options for each subfigure type is set
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        # Initialize the dictionary
        self.defs = {}
        # Default conditions table figure
        self.defs["Conditions"] = {
            "Header": "Conditions",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.4,
            "SkipVars": [],
            "SpecialVars": []
        }
        # Default table of constraints that defines a sweep
        self.defs["SweepConditions"] = {
            "Header": "Sweep Constraints",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.4
        }
        # List of cases in a sweep
        self.defs['SweepCases'] = {
            "Header": "Sweep Cases",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.6
        }
        # Default force/moment table
        self.defs['Summary'] = {
            "Header": "Force \\& moment summary",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.6,
            "Iteration": 0,
            "Components": ["entire"],
            "Coefficients": ["CA", "CY", "CN"],
            "MuFormat": "%.4f",
            "SigmaFormat": "%.4f",
            "EpsFormat": "%.4f",
            "CA": ["mu", "std"],
            "CY": ["mu", "std"],
            "CN": ["mu", "std"],
            "CLL": ["mu", "std"],
            "CLM": ["mu", "std"],
            "CLN": ["mu", "std"]
        }
        # This needs another name, too
        self.defs['ForceTable'] = self.defs['Summary'].copy()
        self.defs['FMTable'] = self.defs['Summary'].copy()
        # Default point sensor table
        self.defs['PointSensorTable'] = {
            "Header": "Point sensor results table",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.6,
            "Iteration": 0,
            "Group": "",
            "Points": [],
            "Targets": [],
            "Coefficients": ["Cp"],
            "Cp": ["mu", "std"],
            "rho": ["mu", "std"],
            "T": ["mu", "std"],
            "p": ["mu", "std"],
            "M": ["mu", "std"],
            "dp": ["mu", "std"]
        }
        # Force or moment iterative history
        self.defs['PlotCoeff'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "Delta": 0.0,
            "StandardDeviation": 0.0,
            "IterativeError": 0.0,
            "ShowMu": [True, False],
            "ShowSigma": [True, False],
            "ShowDelta": [True, False],
            "ShowEpsilon": False,
            "MuFormat": "%.4f",
            "SigmaFormat": "%.4f",
            "DeltaFormat": "%.4f",
            "EpsilonFormat": "%.4f",
            "Format": "pdf",
            "DPI": 150,
            "LineOptions": {"color": ["k", "g", "c", "m", "b", "r"]},
            "MeanOptions": {"ls": None},
            "StDevOptions": {"facecolor": "b", "alpha": 0.35, "ls": "none"},
            "ErrPlotOptions": {
                "facecolor": "g", "alpha": 0.4, "ls": "none"},
            "DeltaPlotOptions": {"color": None},
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Line load plot
        self.defs['PlotLineLoad'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "Format": "pdf",
            "DPI": 150,
            "LineOptions": {
                "color": ["k", "g", "c", "m", "b", "r"]
            },
            "TargetOptions": {
                "color": ["r", "b", "g"],
                "zorder": 2
            },
            "SeamOptions": None,
            "SeamCurves": "smy",
            "SeamLocations": None,
            "Orientation": "vertical",
            "AutoUpdate": True,
            "AdjustLeft": 0.12,
            "AdjustRight": 0.97,
            "AdjustBottom": 0.1,
            "AdjustTop": 0.97,
            "SubplotMargin": 0.015,
            "XPad": 0.03,
            "YPad": 0.03,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Line load plot
        self.defs['SweepLineLoad'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "Format": "pdf",
            "DPI": 150,
            "LineOptions": {
                "color": ["k", "g", "c", "m", "b", "r"]
            },
            "SeamOptions": None,
            "SeamCurves": "smy",
            "SeamLocations": None,
            "Orientation": "vertical",
            "AutoUpdate": False,
            "AdjustLeft": 0.12,
            "AdjustRight": 0.97,
            "AdjustBottom": 0.1,
            "AdjustTop": 0.97,
            "SubplotMargin": 0.015,
            "XPad": 0.03,
            "YPad": 0.03,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Point sensor history
        self.defs['PlotPoint'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Point": 0,
            "Group": "",
            "Coefficient": "Cp",
            "Delta": 0.0,
            "StandardDeviation": 0.0,
            "IterativeError": 0.0,
            "ShowMu": [True, False],
            "ShowSigma": [True, False],
            "ShowDelta": [True, False],
            "ShowEpsilon": False,
            "Format": "pdf",
            "DPI": 150,
            "LineOptions": {
                "color": ["k", "g", "c", "m", "b", "r"],
            },
            "MeanOptions": {"ls": None},
            "StDevOptions": {"facecolor": "b", "alpha": 0.35, "ls": "none"},
            "ErrPlotOptions": {
                "facecolor": "g", "alpha": 0.4, "ls": "none"},
            "DeltaPlotOptions": {"color": None},
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Point sensor retults sweep
        self.defs['SweepPointHist'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "XAxis": None,
            "Target": False,
            "TargetLabel": None,
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Point": 0,
            "Group": "",
            "Coefficient": "Cp",
            "Delta": 0.0,
            "StandardDeviation": 3.0,
            "OutlierSigma": 7.0,
            "Range": 4.0,
            "ShowMu": True,
            "ShowSigma": False,
            "ShowDelta": False,
            "ShowTarget": True,
            "MuFormat": "%.4f",
            "SigmaFormat": "%.4f",
            "DeltaFormat": "%.4f",
            "TargetFormat": "%.4f",
            "Format": "pdf",
            "DPI": 150,
            "PlotMean": True,
            "HistOptions": {"facecolor": "c", "normed": True, "bins": 20},
            "MeanOptions": {"color": "k", "lw": 2},
            "StDevOptions": {"color": "b"},
            "DeltaPlotOptions": {"color": "r", "ls": "--"},
            "TargetOptions": {"color": ["k", "r", "g", "b"], "ls": "--"},
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Force or moment history
        self.defs['SweepCoeff'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "XAxis": None,
            "Target": False,
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "StandardDeviation": 0.0,
            "MinMax": False,
            "LineOptions": {"color": "k", "marker": ["^", "s", "o"]},
            "TargetOptions": {"color": "r", "marker": ["^", "s", "o"]},
            "MinMaxOptions": {
                "facecolor": "g", "color": "g", "alpha": 0.4, "lw": 0.0
            },
            "StDevOptions": {
                "facecolor": "b", "color": "b", "alpha": 0.35, "lw": 0.0
            },
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Histogram of deltas
        self.defs['SweepCoeffHist'] = {
            "HistogramType": "Delta",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "XAxis": None,
            "Target": None,
            "TargetLabel": None,
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "Delta": 0.0,
            "StandardDeviation": 3.0,
            "OutlierSigma": 4.0,
            "Range": 4.0,
            "ShowMu": True,
            "ShowSigma": False,
            "ShowDelta": False,
            "MuFormat": "%.4f",
            "DeltaFormat": "%.4f",
            "SigmaFormat": "%.4f",
            "PlotMean": True,
            "PlotGaussian": False,
            "HistOptions": {"facecolor": "c", "bins": 20},
            "MeanOptions": {"color": "k", "lw": 2},
            "StDevOptions": {"color": "b"},
            "DeltaPlotOptions": {"color": "r", "ls": "--"},
            "GaussianOptions": {"color": "navy", "lw": 1.5},
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Force or moment history
        self.defs['ContourCoeff'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "ContourType": "tricontourf",
            "LineType": "plot",
            "ColorBar": True,
            "XAxis": None,
            "YAxis": None,
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Component": "entire",
            "Coefficient": "CN",
            "LineOptions": {"color": "k", "marker": "o"},
            "ContourOptions": {},
            "AxisEqual": True,
            "ColorMap": "jet",
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Plot L1 residual
        self.defs['PlotL1'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Plot L2 residual
        self.defs['PlotL2'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "YLabel": "L2 residual",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Plot general residual
        self.defs["PlotResid"] = {
            "Residual": "R_1",
            "YLabel": "Residual",
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "FigWidth": 6,
            "FigHeight": 4.5,
            "Format": "pdf",
            "DPI": 150,
            "Grid": None,
            "GridStyle": {},
            "MinorGrid": None,
            "MinorGridStyle": {}
        }
        # Tecplot component 3-view
        self.defs['Tecplot3View'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.66,
            "Component": "entire"
        }
        # General Tecplot layout
        self.defs['Tecplot'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "VarSet": {},
            "ColorMaps": [],
            "FigWidth": 1024,
            "Layout": "layout.lay"
        }
        # Plot a triangulation with Paraview
        self.defs['ParaviewTri'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "Component": "entire",
            "RightAxis": "x",
            "UpAxis": "y"
        }
        # General Paraview script
        self.defs['Paraview'] = {
            "Header": "",
            "Position": "b",
            "Alignment": "center",
            "Width": 0.5,
            "Layout": "layout.py",
            "ImageFile": "export.png",
            "Format": "png",
            "Command": "pvpython"
        }
        self.defs['Image'] = {
            "Header": "",
            "Posittion": "b",
            "Alignment": "center",
            "Width": 0.5,
            "ImageFile": "export.png"
        }

    # Modify defaults or add definitions for a particular module
    def ModSubfigDefaults(self):
        r"""Modify subfigure defaults for a particular solver

        If you are seeing this docstring, then there are no unique
        subfigure defaults for this solver

        :Call:
            >>> opts.ModSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        pass

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

   # --- Category options ---
    # Return all non-default options for a subfigure
    def get_SubfigCascade(self, sfig):
        """Return all options for a subfigure including ones set in a template

        :Call:
            >>> S = opts.get_SubfigCasecasde(sfig)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *S*: :class:`dict`
                Options for subfigure *sfig*
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # get the subfigure options
        S = dict(self.get_Subfigure(sfig))
        # Get the type
        typ = S.get("Type")
        # Exit if not cascading
        if typ == sfig:
            # Self-referenced type
            return S
        # Get list of subfigures
        sfigs = self.get_SubfigList()
        # Check if that type is a template
        if typ not in sfigs:
            # No cascading style
            return S
        # Get the options from that subfigure; recurse
        T = self.get_SubfigCascade(typ)
        # Get new type
        typ = T.get("Type")
        # Overwrite type
        if typ is not None:
            S["Type"] = typ
        # Apply template options but do not overwrite
        for k, v in T.items():
            S.setdefault(k, v)
        # Output
        return S

    # Get the sweep
    def get_Sweep(self, fswp):
        """Return a sweep and its options

        :Call:
            >>> S = opts.get_Sweep(fswp)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fswp*: :class:`str`
                Name of sweep
        :Outputs:
            *S*: :class:`dict`
                Options for sweep *fswp*
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
        # Check for the sweep.
        if fswp in self.get_SweepList():
            # get the sweep.
            return self['Sweeps'][fswp]
        else:
            # Return an empty sweep
            return {}

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
