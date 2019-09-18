"""
:mod:`cape.pycart.options.DataBook`: pyCart DataBook Options
=============================================================

This module provides database options specific to pyCart/Cart3D.  The vast
majority of database options are common to all solvers and are thus inherited
from :class:`cape.options.DataBook.DataBook`.

For force and/or moment components (``"Type": "FM"`` or ``"Type": "Force"``),
each component requested in the databook must also be listed appropriately as a
force and/or moment in the :file:`input.cntl`` file.  These can be written
manually to the template :file:`input.cntl` file or controlled via the
:class:`pyCart.options.Config.Config` class.

The pyCart version of this module alters the default list of columns for
inclusion in the data book.  For point sensors this includes a column called
*RefLev* that specifies the number of refinements of the mesh at the location
of that point sensor (which my vary from case to case depending on mesh
adaptation options).  Point sensors also save the values of state variables at
that point, which for Cart3D are the following columns.

    ==============  ==============================================
    Column          Description
    ==============  ==============================================
    *X*             *x*-coordinate of the point
    *Y*             *y*-coordinate of the point
    *Z*             *z*-coordinate of the point
    *Cp*            Pressure coefficient
    *dp*            :math:`(p-p_\\infty)/(\\gamma p_\\infty)`
    *rho*           Density over freestream density
    *u*             *x*-velocity over freestream sound speed
    *v*             *y*-velocity over freestream sound speed
    *w*             *z*-velocity over freestream sound speed
    *P*             Pressure over gamma times freestream pressure
    ==============  ==============================================
    
The full description of the JSON options can be found in a
:ref:`CAPE section <cape-json-DataBook>` and a 
:ref:`pyCart section <pycart-json-DataBook>`. 

:See Also:
    * :mod:`cape.options.DataBook`
    * :mod:`cape.pycart.options.config.Config`
"""

# Import options-specific utilities
from .util import rc0, getel, odict


# Import base class
import cape.options

# Class for DataBook options
class DataBook(cape.options.DataBook):
    """Dictionary-based interface for DataBook specifications
    
    :Call:
        >>> opts = DataBook(**kw)
    :Outputs:
        *opts*: :class:`pyCart.options.DataBook.DataBook`
            pyCart DataBook options interface
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed from CAPE
    """
            
    # Get the data type of a specific component
    def get_DataBookType(self, comp):
        """Get the type of data book entry for one component
        
        :Call:
            >>> ctype = opts.get_DataBookType(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *ctype*: {Force} | Moment | FM | PointSensor | LineLoad
                Data book entry type
        :Versions:
            * 2015-12-14 ``@ddalle``: First version
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Return the type
        return copts.get("Type", "Force")
    
    # Get additional float columns
    def get_DataBookFloatCols(self, comp):
        """Get additional numeric columns for component (other than coeffs)
        
        This function differs from the standard
        :func:`cape.options.DataBook.DataBook.get_DataBookFloatCols` only in
        that it provides an extra column (*RefLev*) for point and line sensors.
        
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
        elif ctyp in ["PointSensor", "LineSensor"]:
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
        
        For Cart3D point sensors, this also provides the state variables *X*,
        *Y*, *Z*, *Cp*, *dp*, *rho*, *U*, *V*, *W*, *P*.
        
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
        elif ctype in ["PointSensor", "LineSensor"]:
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
                RunMatrix key to use as Mach number or fixed value
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
                RunMatrix key to use as Mach number or fixed value
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
                RunMatrix key to use as Reynolds number or fixed value
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


