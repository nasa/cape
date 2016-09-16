"""Interface for FUN3D data book configuration"""


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
        *opts*: :class:`pyFun.options.DataBook
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed from CAPE
    """
        
    # Get additional float columns
    def get_DataBookFloatCols(self, comp):
        """Get additional numeric columns for component (other than coeffs)
        
        :Call:
            >>> fcols = opts.get_DataBookFloatCols(comp)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component
        :Outputs:
            *fcols*: :class:`list` (:class:`str`)
                List of additional float columns
        :Versions:
            * 2016-09-16 ``@ddalle``: First version
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
        elif ctyp in ['Force', 'Moment', 'FM']:
            # Convergence
            return ["nOrders"]
        else:
            # Global default
            return []
    
    pass

        
# Class for target data
class DBTarget(cape.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


