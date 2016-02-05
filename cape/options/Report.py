"""
Template Module for Automated Report Options 
============================================

This module interfaces options for generating reports.  Since many of the report
options are common to different solvers, much of the report generation content
is controlled here.
"""

# Import options-specific utilities
from .util import rc0, odict, getel, isArray

# Class for flowCart settings
class Report(odict):
    """Dictionary-based interface for options specific to plotting"""
    
    # Initialization method
    def __init__(self, **kw):
        """Initialization method
        
        :Call:
            >>> opts = Report(**kw)
        :Inputs:
            *kw*: :class:`dict` | :class:`cape.options.util.odict`
                Dictionary that is converted to this class
        :Outputs:
            *opts*: :class:`cape.options.Report.Report`
                Options interface
        :Versions:
            * 2016-02-04 ``@ddalle``: First version (not using :class:`dict`)
        """
        # Initialize
        for k in kw:
            self[k] = kw[k]
        # Initialize subfigure defaults
        self.SetSubfigDefaults()
        self.ModSubfigDefaults()
    
    # Subfigure defaults
    def SetSubfigDefaults(self):
        """Set subfigure default options
        
        :Call:
            >>> opts.SetSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Effects:
            *opts.defns*: :class:`dict`
                Default options for each subfigure type is set
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Initialize the dictionary
        self.defs = {}
        # Default conditions table figure
        self.defs["Conditions"] = {
            "Header": "Conditions",
            "Position": "t",
            "Alignment": "left",
            "Width": 0.4,
            "SkipVars": []
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
            "LineOptions": {"color": ["k","g","c","m","b","r"]},
            "MeanOptions": {"ls": None},
            "StDevOptions": {"facecolor": "b", "alpha": 0.35, "ls": "none"},
            "ErrPlotOptions": {
                "facecolor": "g", "alpha": 0.4, "ls": "none"},
            "DeltaOptions": {"color": None}
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
            "LineOptions": {"color": ["k","g","c","m","b","r"]},
            "MeanOptions": {"ls": None},
            "StDevOptions": {"facecolor": "b", "alpha": 0.35, "ls": "none"},
            "ErrPlotOptions": {
                "facecolor": "g", "alpha": 0.4, "ls": "none"},
            "DeltaOptions": {"color": None}
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
            "TargetOptions": {"color": ["k", "r", "g", "b"], "ls": "--"}
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
            "TargetOptions": {"color": "r", "marker": ["^","s","o"]},
            "MinMaxOptions": {
                "facecolor": "g", "color": "g", "alpha": 0.4, "lw": 0.0
            },
            "StDevOptions": {
                "facecolor": "b", "color": "b", "alpha": 0.35, "lw": 0.0
            },
            "Format": "pdf",
            "DPI": 150
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
            "DPI": 150
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
            "DPI": 150
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
            "DPI": 150
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
            "Layout": "layout.py"
        }
        
    # Modify defaults or add definitions for a particular module
    def ModSubfigDefaults(self):
        """Modify subfigure defaults for a particular solver
        
        If you are seeing this docstring, then there are no unique subfigure
        defaults for this solver
        
        :Call:
            >>> opts.ModSubfigDefaults()
        :Inputs:
            *opts*: :class:`cape.options.Report.Report`
                Options interface
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        return None
    
    # List of reports
    def get_ReportList(self):
        """Get list of reports available to create
        
        :Call:
            >>> reps = opts.get_ReportList()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *reps*: :class:`list` (:class:`str`)
                List of reports by name
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the full list of keys.
        K = self.keys()
        # Initialize outputs.
        reps = self.get('Reports', [])
        # Loop through keys/
        for k in K:
            # Check the key
            if k in ['Figures', 'Subfigures', 'Archive', 'Reports']:
                # Known universal option
                continue
            elif k in reps:
                # Already included
                continue
            elif type(self[k]).__name__ != 'dict':
                # Mystery type
                continue
            else:
                # Append to list of reports.
                reps.append(k)
        # Output
        return reps
    
    # List of sweeps
    def get_SweepList(self):
        """Get list of sweeps for a report
        
        :Call:
            >>> fswps = opts.get_SweepList()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of figures by name
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Get sweep list.
        fswps = self.get('Sweeps', {})
        # Output the keys.
        return fswps.keys()
        
    # List of figures (case)
    def get_FigList(self):
        """Get list of figures for a report
        
        :Call:
            >>> figs = opts.get_FigList()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of figures by name
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get figures dictionary.
        figs = self.get('Figures', {})
        # Output the keys.
        return figs.keys()
        
    # List of available subfigures
    def get_SubfigList(self):
        """Get list of available subfigures for a report
        
        :Call:
            >>> figs = opts.get_SubfigList()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *sfigs*: :class:`list` (:class:`str`)
                List of subfigures by name
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get figures dictionary.
        sfigs = self.get('Subfigures', {})
        # Output the keys.
        return sfigs.keys()
        
    # Get the report options.
    def get_Report(self, rep):
        """Return an interface to an individual figure
        
        :Call:
            >>> R = opts.get_Report(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
        :Outputs:
            *R*: :class:`dict`
                Options for figure *rep*
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the figure.
        if rep in self.get_ReportList():
            # Get the figure.
            return self[rep]
        else:
            # Return empty figure.
            return {}
        
    # Get the figure itself.
    def get_Figure(self, fig):
        """Return an interface to an individual figure
        
        :Call:
            >>> F = opts.get_Figure(fig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
        :Outputs:
            *F*: :class:`dict`
                Options for figure *fig*
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the figure.
        if fig in self.get_FigList():
            # Get the figure.
            return self['Figures'][fig]
        else:
            # Return empty figure.
            return {}
        
    # Get the figure itself.
    def get_Subfigure(self, sfig):
        """Return an interface to options for an individual subfigure
        
        :Call:
            >>> S = opts.get_Subfigure(sfig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *S*: :class:`dict`
                Options for subfigure *sfig*
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the figure.
        if sfig in self.get_SubfigList():
            # Get the figure.
            return self['Subfigures'][sfig]
        else:
            # Return empty figure.
            return {}
            
    # Get the sweep
    def get_Sweep(self, fswp):
        """Return a sweep and its options
        
        :Call:
            >>> S = opts.get_Sweep(fswp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fswp*: :class:`str`
                Name of sweep
        :Outputs:
            *S*: :class:`dict`
                Options for sweep *fswp*
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Check for the sweep.
        if fswp in self.get_SweepList():
            # get the sweep.
            return self['Sweeps'][fswp]
        else:
            # Return an empty sweep
            return {}
            
    # Get report list of sweeps.
    def get_ReportSweepList(self, rep):
        """Get list of sweeps in a report
        
        :Call:
            >>> fswps = opts.get_ReportSweepList(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *fswps*: :class:`list` (:class:`str`)
                List of sweeps in the report
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the list of sweeps.
        return R.get('Sweeps', [])
            
    # Get report list of figures.
    def get_ReportFigList(self, rep):
        """Get list of figures in a report
        
        :Call:
            >>> figs = opts.get_ReportFigList(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of figures in the report
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the list of figures.
        return R.get('Figures', [])
        
    # Get report list of figures for cases marked FAIL
    def get_ReportErrorFigList(self, rep):
        """Get list of figures for cases marked FAIL
        
        :Call:
            >>> figs = opts.get_ReportErrorFigList(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of figures in the report
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the list of figures.
        return R.get('ErrorFigures', R.get('Figures', []))
        
    # Get report list of figures for cases marked FAIL
    def get_ReportZeroFigList(self, rep):
        """Get list of figures for cases with zero iterations
        
        :Call:
            >>> figs = opts.get_ReportZeroFigList(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of figures in the report
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the list of figures.
        return R.get('ZeroFigures', R.get('Figures', []))
        
    # Get report title
    def get_ReportTitle(self, rep):
        """Get the title of a report
        
        :Call:
            >>> ttl = opts.get_ReportTitle(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *ttl*: :class:`str`
                Report title
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the title
        return R.get('Title', 'pyCart Automated Report')
        
    # Get report subtitle
    def get_ReportSubtitle(self, rep):
        """Get the subtitle of a report
        
        :Call:
            >>> ttl = opts.get_ReportSubtitle(rep)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *ttl*: :class:`str`
                Report subtitle
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the subtitle
        return R.get('Subtitle', '')
        
    # Get report author
    def get_ReportAuthor(self, rep):
        """Get the title of a report
        
        :Call:
            >>> auth = opts.get_ReportTitle(rep)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *auth*: :class:`str`
                Report author
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the title
        return R.get('Author', '')
        
    # Get report affiliation
    def get_ReportAffiliation(self, rep):
        """Get the author affiliation of a report
        
        :Call:
            >>> afl = opts.get_ReportAffiliation(rep)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *afl*: :class:`str`
                Author affiliation for the report
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the subtitle
        return R.get('Affiliation', '')
        
    # Get report restriction
    def get_ReportRestriction(self, rep):
        """Get the restriction for a report
        
        For example, this may be "SBU - ITAR" or "FOUO"
        
        :Call:
            >>> lbl = opts.get_ReportRestriction(rep)
        :Inputs:
            *opts*: :class:`pycart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *lbl*: :class:`str`
                Distribution restriction
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the title
        return R.get('Restriction', '')
        
    # Get report logo
    def get_ReportLogo(self, rep):
        """Get the file name for the report logo (placed in footer of each page)
        
        :Call:
            >>> fimg = opts.get_ReportLogo(rep)
        :Inputs:
            *opts*: :class:`pycart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *fimg*: :class:`str`
                File name of logo relative to ``report/`` directory
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the title
        return R.get('Logo', '')
        
    # Get report frontispiece (front page logo)
    def get_ReportFrontispiece(self, rep):
        """Get the frontispiece (i.e. title-page logo)
        
        :Call:
            >>> fimg = opts.get_ReportLogo(rep)
        :Inputs:
            *opts*: :class:`pycart.options.Options`
                Options interface
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *fimg*: :class:`str`
                File name of frontispiece relative to ``report/`` directory
        :Versions:
            * 2016-01-29 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Report(rep)
        # Get the title
        return R.get('Frontispiece', '')
        
    # Get report archive status
    def get_ReportArchive(self):
        """Get the option of whether or not to archive report folders
        
        :Call:
            >>> qtar = opts.get_ReportArchive()
        :Inputs:
            *opts*: :class:`pycart.options.Options`
                Options interface
        :Outputs:
            *qtar*: :class:`bool`
                Whether or not to tar archives
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the title
        return self.get('Archive', False)
            
    # Get alignment for a figure
    def get_FigAlignment(self, fig):
        """Get alignment for a figure
        
        :Call:
            >>> algn = opts.get_FigAlignment(fig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
        :Outputs:
            *algn*: :class:`str`
                Figure alignment
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the figure.
        F = self.get_Figure(fig)
        # Get the option
        return F.get('Alignment', 'center')
    
    # Get figure header
    def get_FigHeader(self, fig):
        """Get header (if any) for a figure
        
        :Call:
            >>> lbl = opts.get_FigHeader(fig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
        :Outputs:
            *lbl*: :class:`str`
                Figure header
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the figure.
        F = self.get_Figure(fig)
        # Return the header.
        return F.get('Header', '')
        
    # Get list of figures in a sweep
    def get_SweepFigList(self, fswp):
        """Get list of figures in a sweep
        
        :Call:
            >>> figs = opts.get_SweepFigList(fswp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fswp*: :class:`str`
                Name of sweep
        :Outputs:
            *figs*: :class:`list` (:class:`str`)
                List of "sweep" figures in the report
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Get the report.
        R = self.get_Sweep(fswp)
        # Get the list of figures.
        return R.get('Figures', [])

    # Get list of subfigures in a figure
    def get_FigSubfigList(self, fig):
        """Get list of subfigures for a figure
        
        :Call:
            >>> sfigs = opts.get_FigSubfigList(fig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fig*: :class:`str`
                Name of figure
        :Outputs:
            *sfigs*: :class:`list` (:class:`str`)
                Figure header
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the figure.
        F = self.get_Figure(fig)
        # Return the list of subfigures
        return F.get('Subfigures', [])
        
        
    # Process subfigure type
    def get_SubfigType(self, sfig):
        """Get type for an individual subfigure
        
        :Call:
            >>> t = opts.get_SubfigType(sfig)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *t*: :class:`str`
                Subfigure type
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
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
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
        :Outputs:
            *t*: :class:`str`
                Subfigure parent type
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the subfigure specified type
        t = self.get_SubfigType(sfig)
        # Check if it is a base category.
        if t in ['Conditions', 'SweepConditions', 'SweepCases', 
                'PointSensorTable', 'Summary',
                'PlotCoeff', 'SweepCoeff', 
                'PlotPoint', 'SweepPointHist',
                'PlotL1', 'PlotL2', 'PlotResid',
                'Tecplot3View', 'Tecplot', 'ParaviewTri', 'Paraview']:
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
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
        :Outputs:
            *val*: any
                Sweep option value
        :Versions:
            * 2015-05-28 ``@ddalle``: First version
        """
        # Get the sweep
        S = self.get_Sweep(fswp)
        # Check if the option is present.
        if opt in S:
            # Simple case: option directly specified
            return S[opt]
        # Default values.
        S = {
            "TrajectoryOnly": False,
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
            *opts*: :class:`pyCart.options.Options`
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
            * 2015-03-08 ``@ddalle``: First version
            * 2015-05-22 ``@ddalle``: Support for multiple coeffs in PlotCoeff
        """
        # Get the subfigure.
        S = self.get_Subfigure(sfig)
        # Check if the option is present
        if opt in S:
            # Simple non-default case
            return getel(S[opt], i)
        # Get the type.
        t = self.get_SubfigType(sfig)
        # Process known defaults.
        S = self.defs.get(t)
        # Check for error
        if S is None:
            raise ValueError("Subfigure '%s' type is not recognized" % sfig)
        # Get the default value.
        o = S.get(opt)
        # Process output type.
        return getel(o, i)
        
    # Special function for plot options, which repeat
    def get_SubfigPlotOpt(self, sfig, opt, i):
        """
        Retrieve an option for a subfigure plot, cycling through list of options
        if necessary.
        
        For example, ``{"color": "k", "marker": ["^", "+", "o"]}`` results in a
        sequence of plot options as follows.
        
            0. ``{"color": "k", "marker": "^"}``
            1. ``{"color": "k", "marker": "+"}``
            2. ``{"color": "k", "marker": "o"}``
            3. ``{"color": "k", "marker": "^"}``
            4. ``{"color": "k", "marker": "+"}``
        
        :Call:
            >>> val = opts.get_SubfigPlotOpt(sfig, opt, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
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
            * 2015-06-01 ``@ddalle``: First version
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
# class Report

