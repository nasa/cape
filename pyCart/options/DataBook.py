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
        
    # Get additional float columns
    def get_DataBookFloatCols(self, comp):
        """Get additional numeric columns for component (other than coeffs)
        
        :Call:
            >>> fcols = opts.get_DataBookFloatCols(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component
        :Outputs:
            *fcols*: :class:`list` (:class:`str`)
                List of additional float columns
        :Versions:
            * 2016-03-15 ``@ddalle``: First version
        """
        # Get the component options
        copts = self.get(comp, {})
        # Get type
        ctyp = self.get_DataBookType(comp)
        # Get data book default
        fcols_db = self.get("FloatCols")
        # Get float columns option
        fcols = copts.get("FloatCols")
        # Check for default
        if fcols is not None:
            # Manual option
            return fcols
        elif fcols_db is not None:
            # Data book option
            return fcols_db
        elif ctyp in ["PointSensor"]:
            # Refinement levels for point sensors
            return ["RefLev"]
        elif ctyp in ['Force', 'Moment', 'FM']:
            # Convergence
            return ["nOrders"]
        else:
            # Global default
            return []
        
    # Get the coefficients for a specific component
    def get_DataBookCoeffs(self, comp):
        """Get the list of data book coefficients for a specific component
        
        :Call:
            >>> coeffs = opts.get_DataBookCoeffs(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients for that component
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Check for manually-specified coefficients
        coeffs = copts.get("Coefficients", [])
        # Check the type.
        if type(coeffs).__name__ not in ['list']:
            raise TypeError(
                "Coefficients for component '%s' must be a list." % comp) 
        # Exit if that exists.
        if len(coeffs) > 0:
            return coeffs
        # Check the type.
        ctype = copts.get("Type", "Force")
        # Default coefficients
        if ctype in ["Force", "force"]:
            # Force only, body-frame
            coeffs = ["CA", "CY", "CN"]
        elif ctype in ["Moment", "moment"]:
            # Moment only, body-frame
            coeffs = ["CLL", "CLM", "CLN"]
        elif ctype in ["FM", "full", "Full"]:
            # Force and moment
            coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        elif ctype in ["PointSensor"]:
            # Default to list of points for a point sensor
            coeffs = ["X", "Y", "Z", "Cp", "dp", "rho", "U", "V", "W", "P"]
        # Output
        return coeffs
        
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


