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
        if type(comps).__name__ != 'list':
            comps = [comps]
        # Check contents.
        for comp in comps:
            if not (type(comp).__name__ in ['str', 'int']):
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
        if type(comps).__name__ != 'list':
            comps = [comps]
        # Check contents.
        for comp in comps:
            if not (type(comp).__name__ in ['str', 'int']):
                raise IOError("Component '%s' is not a str or int." % comp)
        # Set the value.
        self.set_key('Components', comps)
        
    # Function to get the coefficients to plot.
    def get_PlotCoeffs(self):
        """Return the list of plot coefficients
        
        :Call:
            >>> coeffs = opts.get_PlotCoeffs()
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
        coeffs = self.get('Coefficients', ['CA', 'CY', 'CN', 'L1'])
        # Make sure it's a list.
        if type(coeffs).__name__ != 'list':
            coeffs = [coeffs]
        # Check contents.
        for coeff in coeffs:
            if not coeff not in ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN', 'L1']:
                raise IOError("Coefficients '%s' not recognized." % coeff)
        # Output
        return coeffs
        
        
        
        
