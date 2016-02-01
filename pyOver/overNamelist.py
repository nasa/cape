"""
Module to interface with "over.namelist" files: :mod:`pyOver.namelist`
======================================================================

This is a module built off of the :mod:`cape.fileCntl` module customized for
manipulating OVERFLOW namelists.
"""

# Import the base file control class.
from cape.namelist2 import Namelist2, np

# Base this class off of the main file control class.
class OverNamelist(Namelist2):
    """
    File control class for :file:`over.namelist`
    ============================================
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> nml = pyOver.Namelist2()
        >>> nml = pyOver.Namelist2(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'over.namelist'``
    :Outputs:
        *nml*: :class:`pyOver.overNamelist.OverNamelist`
            Interface to OVERFLOW input namelist
    :Version:
        * 2016-01-31 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="over.namelist"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.UpdateNamelist()
        # Get grid names
        self.GetGridNames()
        
    # Function to get list of grid names
    def GetGridNames(self):
        """Get the list of grid names in an OVERFLOW namelist
        
        :Call:
            >>> nml.GetGridNames()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get list indices of the 'GRDNAM' lists
        I = self.GetListByName('GRDNAM', None)
        # Save the names as an array (for easier searching)
        self.GridNames = [self.GetKeyFromListIndex(i, 'NAME') for i in I]
        # Save the indices of those lists
        self.iGrid = I
        
    # Get grid number
    def GetGridNumberByName(self, grdnam):
        """Get the number of a grid from its name
        
        :Call:
            >>> i = nml.GetGridNumberByName(grdnam)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
        :Outputs:
            *i*: :class:`int`
                Grid number 
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Check for integer
        if type(grdnam).__name__.startswith('int'):
            return grdnam
        # Check if the grid is present
        if grdnam not in self.GridNames:
            raise KeyError("No grid named '%s' was found" % grdnam)
        # Return the index
        return self.GridNames.index(grdnam)
        
    # Get grid number (alias)
    def GetGridNumber(self, grdnam):
        """Alias of :func:`pyOver.overNamelist.OverNamelist.GetGridNumber`
        
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        return self.GetGridNumberByName(grdnam)
        
    # Get a quantity from a grid (with fallthrough)
    def GetKeyFromGrid(self, grdnam, name, key):
        pass
        
    # Get list of lists in a grid
    def GetListNamesByGridName(self, grdnam):
        """Get the list names in a grid definition
        
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get the start and end indices
        jbeg, jend = self.GetListIndexByGridName
        # Return the corresponding list
        return [self.Names[j] for j in range(jbeg,jend+1)]
    
    # Get start and end of list indices in a grid
    def GetListIndexByGridName(self, grdnam):
        """Get the indices of the first and last list in a grid by name
        
        :Call:
            >>> jbeg, jend = nml.GetListIndexByGridName(grdnam)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
        :Outputs:
            *jbeg*: :class:`int`
                Index of first list in the grid definition
            *jend*: :class:`int`
                Index of last list in the grid definition
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get the grid number
        grdnum = self.GetGridNumberByName(grdnam)
        # Get the list index of this grid's start
        jbeg = self.iGrid[grdnum]
        # Number of grids
        nGrid = len(self.GridNames)
        # Get the list index of the last list in this grid
        if grdnum == nGrid:
            # Use the last list
            jend = len(self.Names)
        else:
            # Use the list before the start of the next grid
            jend = self.iGrid[grdnum+1] - 1
        # Output
        return jbeg, jend
        
        
    # Function set the Mach number.
    def SetMach(self, mach):
        """Set the freestream Mach number
        
        :Call:
            >>> nml.SetMach(mach)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar('reference_physical_properties', 'mach_number', mach)
        
    # Function to get the current Mach number.
    def GetMach(self):
        """
        Find the current Mach number
        
        :Call:
            >>> mach = nml.GetMach()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *M*: :class:`float` (or :class:`str`)
                Mach number specified in :file:`input.cntl`
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
        """
        # Get the value.
        return self.GetVar('reference_physical_properties', 'mach_number')
        
    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        """Set the angle of attack
        
        :Call:
            >>> nml.SetAlpha(alpha)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar('reference_physical_properties', 
            'angle_of_attack', alpha)
        
    # Function to set the sideslip angle
    def SetBeta(self, beta):
        """Set the sideslip angle
        
        :Call:
            >>> nml.SetBeta(beta)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *beta*: :class:`float`
                Sideslip angle
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetVar('reference_physical_properties',
            'angle_of_yaw', beta)
        
    # Set temperature unites
    def SetTemperatureUnits(self, units=None):
        """Set the temperature units
        
        :Call:
            >>> nml.SetTemperatureUnits(units)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *units*: :class:`str`
                Units, defaults to ``"Rankine"``
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        # Check for defaults.
        if units is None: units = "Rankine"
        # Replace the line or add it if necessary.
        self.SetVar('reference_physical_properties',
            'temperature_units', units)
        
    # Set the temperature
    def SetTemperature(self, T):
        """Set the freestream temperature
        
        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *T*: :class:`float`
                Freestream temperature
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'temperature', T)
        
    # Set the Reynolds number
    def SetReynoldsNumber(self, Re):
        """Set the Reynolds number per unit length
        
        :Call:
            >>> nml.SetReynoldsNumber(Re)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *Re*: :class:`float`
                Reynolds number per unit length
        :Versions:
            * 2015-10-15 ``@ddalle``: First version
        """
        self.SetVar('reference_physical_properties', 'reynolds_number', Re)
        
    # Set the number of iterations
    def SetnIter(self, nIter):
        """Set the number of iterations
        
        :Call:
            >>> nml.SetnIter(nIter)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *nIter*: :class:`int`
                Number of iterations to run
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        self.SetVar('code_run_control', 'steps', nIter)
        
    
    # Get the project root name
    def GetRootname(self):
        """Get the project root name
        
        :Call:
            >>> name = nml.GetRootname()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        return self.GetVar('project', 'project_rootname')
    
    
    
    
# class Namelist

        
