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
from ...optdict import OptionsDict, BOOL_TYPES, INT_TYPES


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
    # Attributes
    __slots__ = ()

    # Options
    _optlist = (
        "EqCons",
        "Figures",
        "GlobalCons",
        "MinCases",
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
        "EqCons": str,
        "Figures": str,
        "GlobalCons": str,
        "MinCases": INT_TYPES,
        "TolCons": dict,
        "XCol": str,
        "YCol": str,
    }

    # List depth
    _optlistdepth = {
        "EqCons": 1,
        "Figures": 1,
        "GlobalCons": 1,
    }

    # Defaults
    _rc = {
        "MinCase": 3,
    }

    # Descriptions
    _rst_descriptions = {
        "EqCons": "list of run matrix keys that must be constant on a sweep",
        "Figures": "list of figures in sweep report",
        "GlobalCons": "list of global constraints for sweep",
        "TolCons": "tolerances for run matrix keys to be in same sweep",
        "XCol": "run matrix key to use for *x*-axis of sweep plots",
        "YCol": "run matrix key to use for *y*-axis of sweep contour plots",
    }


# Collection of sweeps
class SweepCollectionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": SweepOpts,
    }


# Class for definition of a figure
class FigureOpts(OptionsDict):
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
    # Additional attibutes
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": FigureOpts,
    }

    # Add Report properties
    @classmethod
    def _add_fig_opts(cls, opts: list, name=None, prefix="Fig"):
        for opt in opts:
            cls._add_fig_opt(opt, name, prefix)

    # Add a property for report
    @classmethod
    def _add_fig_opt(cls, opt: str, name=None, prefix="Fig"):
        r"""Add getter method for ``"Figures"`` option *opt*

        :Call:
            >>> cls._add_fig_opt(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix in method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to getter function
        :Versions:
            * 2023-04-21 ``@ddalle``: v1.0
        """
        # Section subclass
        seccls = FigureOpts
        # Extra args to add
        extra_args = {"fig": (":class:`str`", "figure name")}
        # Default name
        name, fullname = seccls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname

        # Define function
        def func(self, fig: str, i=None, **kw):
            try:
                return self.get_subopt(fig, opt, key="Parent", i=i, **kw)
            except Exception:
                raise

        # Generate docstring
        func.__doc__ = seccls.genr8_getter_docstring(
            opt, name, prefix, extra_args=extra_args)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)


# Add figure getters
FigureCollectionOpts._add_fig_opts(FigureOpts._optlist)


# Class for subfigures
class SubfigOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = (
    )


# Class for subfigure collections
class SubfigCollectionOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": SubfigOpts,
    }

    # Add Report properties
    @classmethod
    def _add_subfig_opts(cls, opts: list, name=None, prefix="Fig"):
        for opt in opts:
            cls._add_subfig_opt(opt, name, prefix)

    # Add a property for report
    @classmethod
    def _add_subfig_opt(cls, opt: str, name=None, prefix="Fig"):
        r"""Add getter method for ``"Subfigures"`` option *opt*

        :Call:
            >>> cls._add_subfig_opt(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix in method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to getter function
        :Versions:
            * 2023-04-21 ``@ddalle``: v1.0
        """
        # Section subclass
        seccls = SubfigOpts
        # Extra args to add
        extra_args = {"sfig": (":class:`str`", "subfigure name")}
        # Default name
        name, fullname = seccls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname

        # Define function
        def func(self, sfig: str, i=None, **kw):
            try:
                return self.get_subopt(sfig, opt, key="Type", i=i, **kw)
            except Exception:
                raise

        # Generate docstring
        func.__doc__ = seccls.genr8_getter_docstring(
            opt, name, prefix, extra_args=extra_args)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)


# Add figure getters
SubfigCollectionOpts._add_fig_opts(FigureOpts._optlist)


# Class for complete *Report* section
class ReportOpts(OptionsDict):
    r"""Dictionary-based interface for automatic report options

    :Call:
        >>> opts = Report(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of archive options
    :Outputs:
        *opts*: :class:`cape.options.Options`
            Automated report options interface
    :Versions:
        * 2016-30-02 ``@ddalle``: v1.0
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
            "DeltaOptions": {"color": None},
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
            "DeltaOptions": {"color": None},
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
            "DeltaOptions": {"color": "r", "ls": "--"},
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
            "DeltaOptions": {"color": "r", "ls": "--"},
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
    # Add Report properties
    @classmethod
    def _add_report_opts(cls, opts: list, name=None, prefix="Report"):
        for opt in opts:
            cls._add_report_opt(opt, name, prefix)

    # Add a property for report
    @classmethod
    def _add_report_opt(cls, opt: str, name=None, prefix="Report"):
        r"""Add getter method for ``"Report"`` option *opt*

        :Call:
            >>> cls._add_report_opt(opt)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix in method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to getter function
        :Versions:
            * 2023-04-20 ``@ddalle``: v1.0
        """
        # Section subclass
        seccls = SingleReportOpts
        # Extra args to add
        extra_args = {"report": (":class:`str`", "report name")}
        # Default name
        name, fullname = seccls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname

        # Define function
        def func(self, report: str, i=None, **kw):
            try:
                return self.get_subopt(report, opt, key="Parent", i=i, **kw)
            except Exception:
                raise

        # Generate docstring
        func.__doc__ = seccls.genr8_getter_docstring(
            opt, name, prefix, extra_args=extra_args)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)

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
    # Process subfigure type
    def get_SubfigType(self, sfig):
        """Get type for an individual subfigure

        :Call:
            >>> t = opts.get_SubfigType(sfig)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *t*: :class:`str`
                Subfigure type
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
        """
        # Get the subfigure
        S = self.get_Subfigure(sfig)
        # Check for a find.
        if S is None:
            raise IOError("Subfigure '%s' was not found." % sfig)
        # Return the type.
        return S.get('Type', '')

    # Get base type of a figure
    def get_SubfigBaseType(self, sfig):
        """Get type for an individual subfigure

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
        # Get the subfigure specified type
        t = self.get_SubfigType(sfig)
        # Check if it is a base category.
        if t in self.defs.keys():
            # Yes, it is.
            return t
        elif t in [sfig, '']:
            # Recursion error
            raise ValueError(
                "Subfigure '%s' does not have recognized type." % sfig)
        else:
            # Derived type; recurse.
            return self.get_SubfigBaseType(t)

    # Get option from a sweep
    def get_SweepOpt(self, fswp, opt):
        """Retrieve an option for a sweep

        :Call:
            >>> val = opts.get_SweepOpt(fswp, opt)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
        :Outputs:
            *val*: any
                Sweep option value
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
        # Get the sweep
        S = self.get_Sweep(fswp)
        # Check if the option is present.
        if opt in S:
            # Simple case: option directly specified
            return S[opt]
        # Default values.
        S = {
            "RunMatrixOnly": False,
            "Figures": [],
            "EqCons": [],
            "TolCons": {},
            "CarpetEqCons": [],
            "CarpetTolCons": {},
            "GlobalCons": [],
            "IndexTol": None,
            "Indices": None,
            "MinCases": 1
        }
        # Output
        return S.get(opt)


    # Process defaults.
    def get_SubfigOpt(self, sfig, opt, i=None, k=None):
        """Retrieve an option for a subfigure, applying necessary defaults

        :Call:
            >>> val = opts.get_SubfigOpt(sfig, opt, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
            *i*: :class:`int`
                Index of subfigure option to extract
        :Outputs:
            *val*: any
                Subfigure option value
        :Versions:
            * 2015-03-08 ``@ddalle``: v1.0
            * 2015-05-22 ``@ddalle``: Support for multiple coeffs in PlotCoeff
        """
        # Ensure *sfig* attribute exists
        try:
            self.sfig
        except AttributeError:
            self.sfig = None
        # Save subfigure name if not set
        if self.sfig is None:
            self.sfig = sfig
        # Get the subfigure.
        S = self.get_Subfigure(sfig)
        # Check if the option is present
        if opt in S:
            # Simple non-default case
            # Remove tag
            self.sfig = None
            # Return option
            return getel(S[opt], i)
        # Get the type.
        t = self.get_SubfigType(sfig)
        # Process known defaults.
        S = self.defs.get(t)
        # Check for error
        if S is None and t == sfig:
            raise ValueError(
                "Subfigure '%s' does not have a recognized type" % self.sfig)
        elif S is None:
            # Cascading type; recurse
            return self.get_SubfigOpt(t, opt, i=i, k=k)
        # Get the default value.
        o = S.get(opt)
        # Delete the original subfigure tag
        self.sfig = None
        # Process output type.
        return getel(o, i)

    # Special function for plot options, which repeat
    def get_SubfigPlotOpt(self, sfig, opt, i):
        """Retrieve an option for a subfigure plot

        For example, ``{"color": "k", "marker": ["^", "+", "o"]}`` results in a
        sequence of plot options as follows.

            0. ``{"color": "k", "marker": "^"}``
            1. ``{"color": "k", "marker": "+"}``
            2. ``{"color": "k", "marker": "o"}``
            3. ``{"color": "k", "marker": "^"}``
            4. ``{"color": "k", "marker": "+"}``

        It is also possible to ...

        :Call:
            >>> val = opts.get_SubfigPlotOpt(sfig, opt, i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
            *i*: :class:`int`
                Index of subfigure option to extract
        :Outputs:
            *val*: any
                Subfigure option value
        :Versions:
            * 2015-06-01 ``@ddalle``: v1.0
        """
        # Get the list of options.
        o_in = self.get_SubfigOpt(sfig, opt)
        # Make sure it's not None
        if o_in is None: o_in = {}
        # Check if it's a list.
        if isArray(o_in) and len(o_in)>0:
            # Cycle through list.
            o_in = o_in[i % len(o_in)]
        # Initialize dict of subfig plot options
        o_plt = {}
        # Loop through keys.
        for k in o_in:
            # Do not apply 'marker' to fill_between plots
            if opt in ['MinMaxOptions', 'StDevOptions', 'ErrPltOptions']:
                if k in ['marker', 'ls']:
                    continue
            # Get the option (may be a list).
            o_k = o_in[k]
            # Check if it's a list.
            if isArray(o_k) and len(o_k)>0:
                # Cycle through the list.
                o_plt[k] = o_k[i % len(o_k)]
            else:
                # Use the non-list value.
                o_plt[k] = o_k
        # Default to the line options if necessary.
        # (This step ensures that StDev and MinMax plots automatically default
        #  to the same color as the Line plots.)
        o_def = self.get_SubfigOpt(sfig, 'LineOptions')
        # Make sure it's not None
        if o_def is None: o_def = {}
        # Check if it's a list.
        if isArray(o_def) and len(o_def)>0:
            # Cycle through list.
            o_def = o_def[i % len(o_def)]
        # Loop through keys.
        for k in o_def:
            # Do not apply 'marker' to fill_between plots
            if opt in ['MinMaxOptions', 'StDevOptions', 'ErrPltOptions']:
                if k in ['marker', 'ls']:
                    continue
            # Get the option (may be a list).
            o_k = o_def[k]
            # Check if it's a list.
            if isArray(o_k) and len(o_k)>0:
                # Cycle through list and set as default.
                o_plt.setdefault(k, o_k[i % len(o_k)])
            else:
                # Use the non-list value as a default.
                o_plt.setdefault(k, o_k)
        # Additional options for area plots
        if opt in ['MinMaxOptions', 'StDevOptions', 'ErrPltOptions']:
            # Check for face color.
            o_plt.setdefault('facecolor', o_plt.get('color'))
        # Output.
        return o_plt


# Add getters for each section
ReportOpts._add_report_opts(SingleReportOpts._optlist, prefix="Report")

