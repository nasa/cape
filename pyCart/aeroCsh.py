"""
Module to interface with :file:`aero.csh` files: :mod:`pyCart.aeroCsh`
======================================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating :file:`aero.csh` files.  The main feature of this module is methods
to set specific properties of the :file:`aero.csh` file, for example the
CFL number or number of adaptation cycles.
"""

# Import the base file control class.
from fileCntl import FileCntl, _num, _float

# Make sure the thing

# Base this class off of the main file control class.
class AeroCsh(FileCntl):
    """
    File control class for :file:`aero.csh` files.
    
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> AC = pyCart.aeroCsh.AeroCsh()
        >>> AC = pyCart.aeroCsh.AeroCsh(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of CNTL file to read, defaults to ``'aero.csh'``
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="aero.csh"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        return None
        
    # Method to write the file.
    def Write(self, fname=None):
        """Write *FC.lines* to text file
        
        :Call:
            >>> FC.Write()
            >>> FC.Write(fname)
        :Inputs:
            *FC*: :class:`pyCart.fileCntl.FileCntl` or derivative
                File control instance, defaults to *FC.fname*
            *fname*: :class:`str`
                Name of file to write to
        :Versions:
            * 2014.06.08 ``@ddalle``: First version
        """
        # Use the FileCntl.WriteEx method instead of FileCntl.Write
        self.WriteEx(fname=fname)
        # Make sure the file is 
        
    # Function to set generic values, since they have the same format.
    def SetVar(self, name, val):
        """Set generic :file:`aero.csh` variable value
        
        :Call:
            >>> AC.SetVar(name, val)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *name*: :class:`str`
                Name of variable as identified in 'aero.csh'
            *val*: *any*, converted using :func:`str`
                Value to which variable is set in final script
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        # Line regular expression: "set XXXX" but with white spaces
        reg = '^\s*set\s+' + str(name) + '\s*[=\n]'
        # Convert *val* to string.
        val = str(val)
        # Form the output line.
        if val == '':
            # Empty line, used to turn a flag off.
            line = 'set %s \n' % name
        else:
            # Set a value.
            line = 'set %s = %s\n' % (name, val)
        # Replace the line; prepend it if missing
        self.ReplaceOrAddLineSearch(reg, line)
        
    # Function to set the functional error tolerance
    def SetErrorTolerance(self, etol):
        """Set error tolerance in :file:`aero.csh` file
        
        :Call:
            >>> AC.SetErrorTolerance(etol)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *etol*: :class:`float`
                Number to set the function error tolerance to
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('etol', etol)
        
    # Function to set the number of refinements
    def SetnRefinements(self, maxR):
        """Set number of refinements for 'cubes' in 'aero.csh' file
        
        :Call:
            >>> AC.SetnRefinements(maxR)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *maxR*: :class:`int`
                Maximum number of refinements for `cubes`
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('maxR', maxR)
        
    # Set the maximum number of cells
    def SetMaxnCells(self, max_nCells):
        """Set the maximum number of cells for the mesh
        
        :Call:
            >>> AC.SetMaxnCells(max_nCells)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *max_nCells*: :class:`int`
                Maximum number of cells allowed in mesh
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('max_nCells', max_nCells)
        
    # Number of adaptation cycles.
    def SetnAdapt(self, n_adapt_cycles):
        """Set the number of adaptation cycles
        
        :Call:
            >>> AC.SetnAdapt(n_adapt_cycles)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *n_adapt_cycles*: :class:`int`
                Number of adaptation cycles
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('n_adapt_cycles', n_adapt_cycles)
        
    # Number of flowCart iterations on initial mesh
    def SetnIter(self, it_fc):
        """Set the *initial* number of flowCart iterations
        
        :Call:
            >>> AC.SetnIter(it_fc)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *it_fc*: :class:`int`
                Number of flowCart iters on initial mesh
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('it_fc', it_fc)
        
    # Number of flowCart iterations on subsequent meshes
    def SetnIterList(self, ws_it):
        """Set the number of flowCart iterations on new mesh
        
        :Call:
            >>> AC.SetnIterList(ws_it)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *ws_it*: :class:`int`
                Number of `flowCart` iters on subsequent meshes
        :Effects:
            Writes a line of the form ``'set ws_it = ( 50 50 50 )'``
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        # Number of values specified
        n_ws = len(ws_it)
        # Initialize the string.
        line = '('
        # Loop through values.
        for n in ws_it:
            # Add a space and the number
            line += ' %s' % n
        # Write at least 12 entries.
        if n_ws < 12:
            # Repeat the last value until there are 12 entries.
            line += (' %s' % n)*(12-n_ws)
        # Finish the line.
        line += ' )'
        # Replace it.
        self.SetVar('ws_it', line)
        
    # Number of adjointCart iterations
    def SetnIterAdjoint(self, it_ad):
        """Set the number of adjointCart iterations
        
        :Call:
            >>> AC.SetnIter(it_ad)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *it_ad*: :class:`int`
                Number of `adjointCart` iters
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('it_ad', it_ad)
        
    # Set CFL number
    def SetCFL(self, cfl):
        """Set the CFL number
        
        :Call:
            >>> AC.SetCFL(cfl)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *cfl*: :class:`float`
                CFL number
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        self.SetVar('cfl', cfl)
        
    # Set min CFL number
    def SetCFLMin(self, cflmin):
        """Set the minimum CFL number, which :file:`aero.csh` uses as backup
        
        :Call:
            >>> AC.SetCFLMin(cflmin)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *cfl*: :class:`float`
                CFL number
        :Versions:
            * 2014.10.03 ``@ddalle``: First version
        """
        self.SetVar('cflmin', cflmin)
        
    # Set limiter type number
    def SetLimiter(self, limiter):
        """Set the limiter for `flowCart`
        
        :Call:
            >>> AC.SetLimiter(limiter)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *limiter*: :class:`int`
                Limiter, 2 or 5
        :Versions:
            * 2014.10.13 ``@ddalle``: First version
        """
        self.SetVar('limiter', int(limiter))
        
    # Set the number of multigrid levels
    def SetnMultiGrid(self, mg_fc):
        """Set the number of multigrid levels for both solvers
        
        :Call:
            >>> AC.SetnMultiGrid(mg_fc)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *mg_fc*: :class:`int`
                Number of multigrid levels, applied to flowCart and adjointCart
        """
        # Versions:
        #  2014.06.10 @ddalle  : First version
        
        # Set both variables.
        self.SetVar('mg_fc', mg_fc)
        self.SetVar('mg_ad', mg_fc)
        
    # Set the mesh growth factor list
    def SetMeshGrowth(self, mesh_growth):
        """Set the list of mesh growth factors
        
        :Call:
            >>> AC.SetMeshGrowth(mesh_growth)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *mesh_growth*: *array_like*
                Vector of mesh growth parameters
        :Versions:
            * 2014.06.10 ``@ddalle``: First version
        """
        # Number of values specified
        n_ws = len(mesh_growth)
        # Initialize the string.
        line = '('
        # Loop through values.
        for n in mesh_growth:
            # Add a space and the number
            line += ' %s' % float(n)
        # Write at least 12 entries.
        if n_ws < 12:
            # Repeat the last value until there are 12 entries.
            line += (' %s' % float(n))*(12-n_ws)
        # Finish the line.
        line += ' )'
        # Replace it.
        self.SetVar('mesh_growth', line)
        
    # Set the mesh growth method
    def SetAPC(self, apc):
        """Set the list of mesh growth factors
        
        :Call:
            >>> AC.SetAPC(apc)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *apc*: *array_like*
                Vector of ``'p'`` and ``'a'``
        :Versions:
            * 2014.10.13 ``@ddalle``: First version
        """
        # Number of values specified
        n_ws = len(apc)
        # Initialize the string.
        line = '('
        # Loop through values.
        for c in apc:
            # Add a space and the number
            line += (" " + c)
        # Write at least 12 entries.
        if n_ws < 12:
            # Repeat the last value until there are 12 entries.
            line += (" "+c)*(12-n_ws)
        # Finish the line.
        line += ' )'
        # Replace it.
        self.SetVar('apc', line)
        
    # Set y_is_spanwise on or off.
    def SetYIsSpanwise(self, y_is_spanwise):
        """Turn on or off *y_is_spanwise* flag
        
        :Call:
            >>> AC.SetYIsSpanwise(y_is_spanwise)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *y_is_spanwise*: :class:`bool`
                Whether or not to use *y*-axis as spanwise axis
        :Versions:
            * 2014.11.11 ``@ddalle``: First version
        """
        # Check value.
        if y_is_spanwise:
            # Set the flag.
            val = '-y_is_spanwise'
        else:
            # Set the flag to empty.
            val = ''
        # Modify the line to its appropriate value.
        self.SetVar('y_is_spanwise', val)
        
    # Set y_is_spanwise on or off.
    def SetFMG(self, fmg):
        """Turn on or off ``-no_fmg`` flag
        
        :Call:
            >>> AC.SetFMG(fmg)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *fmg*: :class:`bool`
                Whether or not to use full multigrid
        :Versions:
            * 2014-11-13 ``@ddalle``: First version
        """
        # Check value.
        if fmg:
            # Set the flag to empty.
            val = ''
        else:
            # Set the -no_fmg flag.
            val = '-no_fmg'
        # Modify the line to its appropriate value.
        self.SetVar('fmg', val)
        
    # Set y_is_spanwise on or off.
    def SetPMG(self, pmg):
        """Turn on or off ``-pmg`` flag
        
        :Call:
            >>> AC.SetPMG(pmg)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *pmg*: :class:`bool`
                Whether or not to use poly multigrid
        :Versions:
            * 2014-11-13 ``@ddalle``: First version
        """
        # Check value.
        if pmg:
            # Turn the pmg flag on
            val = '-pmg'
        else:
            # No PMG flag
            val = ''
        # Modify the line to its appropriate value.
        self.SetVar('pmg', val)
    
    # Function to set the number of buffers
    def SetABuffer(self, buf):
        """Set the number of buffer layers when refining a cell
        
        :Call:
            >>> AC.SetABuffer(buf)
        :Inputs:
            *AC*: :class:`pyCart.aeroCsh.AeroCsh`
                Instance of the :file:`aero.csh` manipulation class
            *buf*: :class:`int`
                Number of buffer layers
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        # Set this value (integer)
        self.SetVar('buf', buf)
    
    
    
    
    
    
    
    
