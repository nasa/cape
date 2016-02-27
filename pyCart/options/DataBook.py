"""Interface for Cart3D data book configuration"""


# Import options-specific utilities
from util import rc0, getel, odict


# Import base class
import cape.options

# Class for DataBook options
class DataBook(cape.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyCart.options.DataBook
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed from CAPE
    """
        
    # Get the Mach number option
    def get_ComponentMach(self, comp):
        """Get the Mach number option for a data book group
        
        Mostly used for line loads; coefficients require Mach number
        
        :Call:
            >>> o_mach = opts.get_ComponentMach(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load group
        :Outputs:
            *o_mach*: :class:`str` | :class:`float`
                Trajectory key to use as Mach number or fixed value
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Get component options
        copts = self.get(comp, {})
        # Get the Mach number option
        return self.get("Mach", "mach")
        
    # Get the gamma option
    def get_ComponentGamma(self, comp):
        """Get the Mach number option for a data book group
        
        Mostly used for line loads; coefficients require Mach number
        
        :Call:
            >>> o_mach = opts.get_ComponentMach(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load group
        :Outputs:
            *o_mach*: :class:`str` | :class:`float`
                Trajectory key to use as Mach number or fixed value
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Get component options
        copts = self.get(comp, {})
        # Get the Mach number option
        return self.get("Gamma", 1.4)
        
    # Get the Reynolds Number option
    def get_ComponentReynoldsNumber(self, comp):
        """Get the Reynolds number option for a data book group
        
        Mostly used for line loads; coefficients require Mach number
        
        :Call:
            >>> o_re = opts.get_ComponentReynoldsNumber(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of line load group
        :Outputs:
            *o_mach*: :class:`str` | :class:`float`
                Trajectory key to use as Reynolds number or fixed value
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Get component options
        copts = self.get(comp, {})
        # Get the Mach number option
        return self.get("Re", None)
        
        
        
# Class for target data
class DBTarget(cape.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


