"""
This module contains the interface for data book options for pyFun and FUN3D.
The classes in this module are subclassed as
    
    * :class:`pyFun.options.DataBook.DataBook` ->
      :class:`cape.cfdx.options.DataBook.DataBook`
      
    * :class:`pyFun.options.DataBook.DBTarget` ->
      :class:`cape.cfdx.options.DataBook.DBTarget`

The FUN3D-specific options for these classes are almost null, but a few methods
are modified in order to change default data book component types and the
columns of data available for each.

"""


# Import options-specific utilities
from .util import rc0, getel, odict


# Import base class
import cape.cfdx.options

# Class for DataBook options
class DataBook(cape.cfdx.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyFun.options.DataBook`
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed from CAPE
    """
            
    # Get the data type of a specific component
    def get_DataBookType(self, comp):
        """Get the type of data book entry for one component
        
        :Call:
            >>> ctype = opts.get_DataBookType(comp)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *ctype*: {Force} | Moment | FM | PointSensor | LineLoad
                Data book entry type
        :Versions:
            * 2015-12-14 ``@ddalle``: First version
            * 2017-04-04 ``@ddalle``: Overwriting cape's default
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Return the type
        return copts.get("Type", "FM")

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
            *fcols*: :class:`list`\ [:class:`str`]
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
class DBTarget(cape.cfdx.options.DBTarget):
    """Dictionary-based interface for databook targets"""
    
    pass


