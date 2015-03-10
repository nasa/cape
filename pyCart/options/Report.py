"""Interface for pyCart automated report generation"""


# Import options-specific utilities
from .util import rc0, odict

# Class for flowCart settings
class Report(odict):
    """Dictionary-based interface for options specific to plotting"""
    
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
        reps = []
        # Loop through keys/
        for k in K:
            # Check the key
            if k in ['Figures', 'Subfigures', 'Archive']:
                # Known universal option
                continue
            elif type(self[k]).__name__ != 'dict':
                # Mystery type
                continue
            else:
                # Append to list of reports.
                reps.append(k)
        # Output
        return reps
            
    # List of figures
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
            return
            
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
        """Get the restriction for a report
        
        For example, this may be "SBU - ITAR" or "FOUO"
        
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
            *t*: ['Conditions', 'Summary', 'PlotCoeff', 'PlotL1']
                Subfigure parent type
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the subfigure specified type
        t = self.get_SubfigType(sfig)
        # Check if it is a base category.
        if t in ['Conditions', 'Summary', 'PlotCoeff', 'PlotL1',
        'Tecplot3View']:
            # Yes, it is.
            return t
        else:
            # Derived type; recurse.
            return self.get_SubfigBaseType(t)
        
    # Process defaults.
    def get_SubfigOpt(self, sfig, opt):
        """Retrieve an option for a subfigure, applying necessary defaults
        
        :Call:
            >>> val = opts.get_SubfigOpt(sfig, opt)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sfig*: :class:`str`
                Name of subfigure
            *opt*: :class:`str`
                Name of option to retrieve
        :Outputs:
            *val*: any
                Subfigure option value
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Get the subfigure.
        S = self.get_Subfigure(sfig)
        # Check if the option is present
        if opt in S:
            # Simple non-default case
            return S[opt]
        # Get the type.
        t = self.get_SubfigType(sfig)
        # Process known defaults.
        if t in ['Conditions']:
            # Default conditions subfigure
            S = {
                "Header": "Conditions",
                "Position": "t",
                "Alignment": "left",
                "Width": 0.4,
                "SkipVars": []
            }
        elif t in ['Summary']:
            # Default results summary
            S = {
                "Header": "Force \\& moment summary",
                "Position": "t",
                "Alignment": "left",
                "Width": 0.6,
                "Iteration": 0,
                "Components": ["entire"],
                "Coefficients": ["CA", "CY", "CN"],
                "CA": ["mu", "std", "err"],
                "CY": ["mu", "std", "err"],
                "CN": ["mu", "std", "err"],
                "CLL": ["mu", "std", "err"],
                "CLM": ["mu", "std", "err"],
                "CLN": ["mu", "std", "err"]
            }
        elif t in ['PlotCoeff']:
            # Force or moment history
            S = {
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
                "Format": "pdf",
                "DPI": 150
            }
        elif t in ['PlotL1']:
            # Residual history
            S = {
                "Header": "",
                "Position": "b",
                "Alignment": "center",
                "Width": 0.5,
                "FigWidth": 6,
                "FigHeight": 4.5,
                "Format": "pdf",
                "DPI": 150
            }
        elif t in ['Tecplot3View']:
            # Component 3-view
            S = {
                "Header": "",
                "Position": "b",
                "Alignment": "center",
                "Width": 0.66,
                "Component": "entire"
            }
        else:
            # This is a derived subfigure type; recurse.
            return self.get_SubfigOpt(t, opt)
        # Get the default value.
        return S.get(opt)
        
        
