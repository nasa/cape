"""
Module to interface with "input.cntl" files: :mod:`pyCart.inputCntl`
====================================================================

This is a module built off of the :mod:`pyCart.fileCntl` module customized for
manipulating :file:`input.cntl` files.  Such files are split into section by lines of
the format

    ``$__Post_Processing``
    
and this module is designed to recognize such sections.  The main feature of
this module is methods to set specific properties of the :file:`input.cntl` 
file, for example the Mach number or CFL number.
"""

# Import the base file control class.
import cape.namelist

# Base this class off of the main file control class.
class Namelist(cape.namelist.Namelist):
    """
    File control class for :file:`fun3d.nml`
    ========================================
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> nml = pyFun.Namelist()
        >>> nml = pyfun.Namelist(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'fun3d.nml'``
    :Version:
        * 2015-10-15 ``@ddalle``: Started
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="fun3d.nml"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.SplitToSections(reg="\&([\w_]+)")
        
    # Set restart on
    def SetRestart(self, q=True):
        """Set the FUN3D restart flag on or off
        
        :Call:
            >>> nml.SetRestart(q=True)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *q*: :class:`bool` | ``"on"`` | ``"off"`` | ``None``
                Restart option, ``None`` turns flag to ``"on"``
        :Versions:
            * 2015-11-03 ``@ddalle``: First version
        """
        # Check status
        if (q is None) or (q and (q != "off")):
            # Turn restart on.
            self.SetVar('code_run_control', 'restart_read', 'on')
        else:
            # Turn restart off.
            self.SetVar('code_run_control', 'restart_read', 'off')
        
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
    
    # Set the project root name
    def SetRootname(self, name):
        """Set the project root name
        
        :Call:
            >>> nml.SetRootname(name)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *name*: :class:`str`
                Name of project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        self.SetVar('project', 'project_rootname', name)
    
    # Get the adapt project root name
    def GetAdaptRootname(self):
        """Get the post-adaptation project root name
        
        :Call:
            >>> name = nml.GetAdaptRootname()
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
        :Outputs:
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        return self.GetVar('adapt_mechanics', 'adapt_project')
        
    # Set the adapt project root name
    def SetAdaptRootname(self, name):
        """Set the post-adaptation project root name
        
        :Call:
            >>> nml.SetAdaptRootname(name)
        :Inputs:
            *nml*: :class:`pyFun.namelist.Namelist`
                File control instance for :file:`fun3d.nml`
            *name*: :class:`str`
                Name of adapted project
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        self.SetVar('adapt_mechanics', 'adapt_project', name)
        
# class Namelist

        
