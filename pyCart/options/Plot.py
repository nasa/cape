"""Interface for options specific to plotting"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class Plot(odict):
    """Dictionary-based interface for options specific to plotting"""
    
    # List of components to plot
    def get_PlotComponents(self):
        """Return the list of components to plot
        
        :Call:
            >>> comps = opts.get_PlotComponents()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *comps*: :class:`list` (:class:`str` | :class:`int`)
                List of components (names or numbers) to plot
        :Versions:
            * 2014-11-22 ``@ddalle``: First version
        """
        # Get the value from the dictionary.
        comps = self.get('Components', ['entire'])
        # Make sure it's a list.
        if type(comps).__name__ not in ['list']:
            comps = [comps]
        # Check contents.
        for comp in comps:
            if (type(comp).__name__ not in ['str', 'int', 'unicode']):
                raise IOError("Component '%s' is not a str or int." % comp)
        # Output
        return comps
        
    # Set run input sequence.
    def set_PlotComponents(self, comps):
        """Set the list of components to plot
        
        :Call:
            >>> opts.set_PlotComponents(comps)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comps*: :class:`list` (:class:`str` | :class:`int`)
                List of components (names or numbers) to plot
        :Versions:
            * 2014-11-22 ``@ddalle``: First version
        """
        # Make sure it's a list.
        if type(comps).__name__ not in ['list']:
            comps = [comps]
        # Check contents.
        for comp in comps:
            if (type(comp).__name__ not in ['str', 'int', 'unicode']):
                raise IOError("Component '%s' is not a str or int." % comp)
        # Set the value.
        self.set_key('Components', comps)
        
    # Function to add to the list of components.
    def add_PlotComponents(self, comps):
        """Add to the list of components to plot
        
        :Call:
            >>> opts.add_PlotComponents(comps)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comps*: :class:`list` (:class:`str` | :class:`int`)
                List of components (names or numbers) to plot
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the current list.
        comps_cur = self.get('Components', [])
        # Make sure it's a list.
        if type(comps_cur).__name__ not in ['list']:
            comps_cur = [comps_cur]
        # Check the type of the input.
        try:
            # Try it as a list first.
            comps_cur += comps
        except Exception:
            # Append it as a string/int.
            comps_cur.append(comps)
        # Set the value.
        self['Components'] = comps_cur
        
        
    # Function to get the coefficients to plot.
    def get_PlotCoeffs(self, comp=None):
        """Return the list of plot coefficients for a component
        
        This applies the default from the "Plot" section of the options, but
        this is superseded by specific exceptions.
        
        :Call:
            >>> coeffs = opts.get_PlotCoeffs(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients to plot
        :Versions:
            * 2014-11-22 ``@ddalle``: First version
        """
        # Get the value from the dictionary.
        coeffs = self.get('Coefficients', ['CA', 'CY', 'CN', 'L1'])
        # Check for a specific coefficient.
        if comp in self:
            # List from specific coefficient supersedes.
            coeffs = self[comp].get('Coefficients', coeffs)
        # Make sure it's a list.
        if type(coeffs).__name__ not in ['list']:
            coeffs = [coeffs]
        # Check contents.
        for coeff in coeffs:
            if coeff not in ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN', 'L1',
                    'CAhist', 'CYhist', 'CNhist',
                    'CLLhist', 'CLMhist', 'CLNhist']:
                raise IOError("Coefficient '%s' not recognized." % coeff)
        # Output
        return coeffs
        
        
    # Function to get the number of iterations
    def get_nPlotIter(self, comp=None):
        """Return the number of iterations to plot for a component
        
        If there are fewer than *nPlot* iterations in the current history, all
        iterations will be plotted.
        
        :Call:
            >>> nPlot = opts.get_nPlotIter(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *nPlot*: :class:`int`
                Number of iterations to plot (maximum)
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the default.
        nPlot = self.get('nPlot', rc0('nPlot'))
        # Check for specific component.
        if comp in self:
            # Value supersedes
            nPlot = self[comp].get('nPlot', nPlot)
        # Output
        return nPlot
        
    # Function to get the number of iterations for averaging
    def get_nAverage(self, comp=None):
        """Return the number of iterations to use for averaging
        
        If there are fewer than *nAvg* iterations in the current history, all
        iterations will be plotted.
        
        :Call:
            >>> nAvg = opts.get_nAverage()
            >>> nAvg = opts.get_nAverage(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *nAvg*: :class:`int`
                Number of iterations to use for averaging
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the default.
        nAvg = self.get('nAverage', rc0('nAvg'))
        # Check for specific component to supersede.
        if comp in self:
            nAvg = self[comp].get('nAverage', nAvg)
        # Output
        return nAvg
        
    # Function to get the number of rows of plots
    def get_nPlotRows(self, comp=None):
        """Return the number of rows to use in plots
        
        :Call:
            >>> nRow = opts.get_nPlotRows()
            >>> nRow = opts.get_nPlotRows(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *nRow*: :class:`int`
                Number of rows of plots
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the default.
        nRow = self.get('nRow', rc0('nRow'))
        # Check for specific component to supersede
        if comp in self:
            nRow = self[comp].get('nRow', nRow)
        # Output
        return nRow
        
    # Function to get the number of columns of plots
    def get_nPlotCols(self, comp=None):
        """Return the number of columns to use in plots
        
        :Call:
            >>> nCol = opts.get_nPlotCols()
            >>> nCol = opts.get_nPlotCols(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *nCol*: :class:`int`
                Number of columns of plots
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the default.
        nCol = self.get('nCol', rc0('nCol'))
        # Check for specific component to supersede
        if comp in self:
            nCol = self[comp].get('nCol', nCol)
        # Output            
        return nCol
        
    # Function to get the number of columns of plots
    def get_PlotRestriction(self, comp=None):
        """Return the number of columns to use in plots
        
        :Call:
            >>> sTag = opts.get_nPlotRestriction()
            >>> sTag = opts.get_nPlotRestriction(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
        :Outputs:
            *sTag*: :
                Number of columns of plots
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Get the default.
        sTag = self.get('Restriction', '')
        # Check for specific component to supersede
        if comp in self:
            sTag = self[comp].get('Restriction', sTag)
        # Output
        return sTag
        
    # Function to get the delta for a given comp and coeff
    def get_PlotDelta(self, coeff, comp=None):
        """
        Get the fixed-width interval to plot above and below the mean for a
        specific component and coefficient
        
        :Call:
            >>> dC = opts.get_PlotDelta(coeff, comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component to plot
            *coeff*: :class:`str`
                Name of coefficient 
        :Outputs:
            *dC*: :class:`float` or ``None``
                Offset to plot from mean with dotted line
        :Versions:
            * 2014-11-23 ``@ddalle``: First version
        """
        # Check for recognized coefficient
        if coeff not in ['CA','CY','CN','CLL','CLM','CLN']:
            # Null output
            return None
        # Check for a hard default
        dC = self.get('d'+coeff, rc0('dC'))
        # Get the list of Deltas
        Deltas = self.get('Deltas', {})
        # Process the correct universal value.
        dC = Deltas.get(coeff, dC)
        # Check for component-specific information
        if comp not in self: return dC
        # Check for hard value for the component.
        dC = self[comp].get('d'+coeff, dC)
        # Check for dictionary
        Deltas = self[comp].get('Deltas', {})
        # Get the value from that dictionary.
        dC = Deltas.get(coeff, dC)
        # Output
        return dC
                
        
