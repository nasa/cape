"""Interface for pyCart automated report generation"""


# Import options-specific utilities
from .util import rc0, odict

# Class for flowCart settings
class Report(odict):
    """Dictionary-based interface for options specific to plotting"""
    
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
            *F*: :class:`pyCart.options.Report.Figure` or :class:`dict`
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
            *S*: :class:`pyCart.options.Report.Subfigure` or :class:`dict`
                Options for subfigure *sfig*
        :Versions:
            * 2015-03-08 ``@ddalle``: First version
        """
        # Check for the figure.
        if fig in self.get_SubfigList():
            # Get the figure.
            return self['Subfigures'][sfig]
        else:
            # Return empty figure.
            return 
            
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
    


    # Get list of subfigures in a figure
    
    
