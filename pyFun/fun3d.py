"""
FUN3D control module: :mod:`pyFun.fun3d`
========================================

This module provides tools to interface with most FUN3D settings from Python.
"""

# Basic numerics
import numpy as np
# Configuration file processor
import json
# Date processing
from datetime import datetime
# File system and operating system management
import os, shutil
import subprocess as sp

# Import template class
from cape.cntl import Cntl

# Local classes
from namelist  import Namelist

# Other pyFun modules
from . import options
from . import case
# Unmodified CAPE modules
from cape import convert

# Functions and classes from other modules
from trajectory import Trajectory

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(_fname)[0]
    
# Class to read input files
class Fun3d(Cntl):
    """
    Class for handling global options and setup for FUN3D.
    
    This class is intended to handle all settings used to describe a group
    of Cart3D cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha sweep
    for each single control variable setting.
    
    The settings are read from a JSON file, which is robust and simple to
    read, but has the disadvantage that there is no support for comments.
    Hopefully the various names are descriptive enough not to require
    explanation.
    
    Defaults are read from the file ``$CAPE/settings/pyFun.default.json``.
    
    :Call:
        >>> fun3d = pyFun.Fun3d(fname="pyFun.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyFun input file
    :Outputs:
        *fun3d*: :class:`pyFun.fun3d.Fun3d`
            Instance of the pyFun control class
    :Data members:
        *fun3d.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *fun3d.x*: :class:`pyFun.trajectory.Trajectory`
            Values and definitions for variables in the run matrix
        *fun3d.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: Started
    """ 
    # Initialization method
    def __init__(self, fname="pyFun.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.ImportModules()
        
        # Process the trajectory.
        self.x = Trajectory(**self.opts['Trajectory'])

        # Job list
        self.jobs = {}
        
        # Read the namelist.
        self.ReadNamelist()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyFun.Fun3d(nCase=%i)>" % (
            self.x.nCase)
        
        
        
    # Read the namelist
    def ReadNamelist(self, j=0):
        """Read the :file:`fun3d.nml` file
        
        :Call:
            >>> fun3d.ReadInputCntl(j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of the pyFun control class
            *j*: :class:`int`
                Run sequence index
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # CHange to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file.
        self.Namelist = Namelist(self.opts.get_Namelist(j))
        # Go back to original location
        os.chdir(fpwd)
        
    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        """Get a namelist variable's value
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> val = fun3d.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Get the namelist value.
        nval = self.Namelist.GetVar(sec, key)
        # Check for options value.
        if nval is None:
            # No namelist file value
            return self.opts.get_namelist_var(sec, key, j)
        elif 'Fun3D' not in self.opts:
            # No namelist in options
            return nval
        elif sec not in self.opts['Fun3D']:
            # No corresponding options section
            return nval
        elif key not in self.opts['Fun3D'][sec]:
            # Value not specified in the options namelist
            return nval
        else:
            # Default to the options
            return self.opts_get_namelist_var(sec, key, i)
        
    # Get the project rootname
    def GetProjectRootName(self, j=0):
        """Get the project root name
        
        The JSON file overrides the value from the namelist file if appropriate
        
        :Call:
            >>> name = fun3d.GetProjectName(j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Read the namelist.
        self.ReadNamelist(j)
        # Get the namelist value.
        nname = self.Namelist.GetVar('project', 'project_rootname')
        # Check for options value
        if nfmt is None:
            # Use the options value.
            return self.opts.get_project_rootname(j)
        elif 'Fun3D' not in self.opts:
            # No namelist options
            return nname
        elif 'project' not in self.opts['Fun3D']:
            # No project options
            return nname
        elif 'project_rootname' not in self.opts['Fun3D']['project']:
            # No rootname
            return nname
        else:
            # Use the options value.
            return self.opts.get_project_rootname(j)
            
    # Get the grid format
    def GetGridFormat(self, j=0):
        """Get the grid format
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> fmt = fun3d.GetGridFormat(j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *fmt*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        return self.GetNamelistVar('raw_grid', 'grid_format', j)
            

    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> n = fun3d.CaseGetCurrentIter()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.GetCurrentIter()
        
    # Get last iter
    def GetLastIter(self, i):
        """Get minimum required iteration for a given run to be completed
        
        :Call:
            >>> nIter = fun3d.GetLastIter(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        # Check the case
        if self.CheckCase(i) is None:
            return None
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Read the local case.json file.
        rc = case.ReadCaseJSON()
        # Return to original location.
        os.chdir(fpwd)
        # Output
        return rc.get_LastIter()
        
        
        
        
    # Function to prepare "input.cntl" files
    def PrepareNamelist(self, i):
        """
        Write :file:`fun3d.nml` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> fun3d.PrepareNamelist(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
            * 2014-06-06 ``@ddalle``: Low-level functionality for grid folders
            * 2014-09-30 ``@ddalle``: Changed to write only a single case
        """
        # Read namelist file
        self.ReadNamelist()
        # Extract trajectory.
        x = self.x
        # Process the key types.
        KeyTypes = [x.defns[k]['Type'] for k in x.keys]
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Set the flight conditions.
        # Mach number
        for k in x.GetKeysByType('Mach'):
            self.Namelist.SetMach(getattr(x,k)[i])
        # Angle of attack
        if 'alpha' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('alpha')[0]
            # Set the value.
            self.Namelist.SetAlpha(getattr(x,k)[i])
        # Sideslip angle
        if 'beta' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('beta')[0]
            # Set the value.
            self.Namelist.SetBeta(getattr(x,k)[i])
        # Check for total angle of attack.
        if 'alpha_t' in KeyTypes:
            # Find out which key it is.
            k = x.GetKeysByType('alpha_t')[0]
            # Get the value.
            av = getattr(x,k)[i]
            # Check for roll angle.
            if 'phi' in KeyTypes:
                # Kind the ky.
                k = x.GetKeysByType('phi')[0]
                # Get the value.
                rv = getattr(x,k)[i]
            else:
                # Set roll to zero.
                rv = 0.0
            # Convert the values to aoa and aos.
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Set them.
            self.Namelist.SetAlpha(a)
            self.Namelist.SetBeta(b)
        ## Specify list of forces to track with `clic`
        #self.Namelist.RequestForce(self.opts.get_ClicForces())
        ## Set reference values.
        #self.Namelist.SetReferenceArea(self.opts.get_RefArea())
        #self.Namelist.SetReferenceLength(self.opts.get_RefLength())
        #self.Namelist.SetMomentPoint(self.opts.get_RefPoint())
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Loop through input sequence
        for j in range(self.opts.get_nSeq()):
            # Get the reduced namelist for sequence *j*
            nopts = self.opts.select_namelist(j)
            # Apply them to this namelist
            self.Namelist.ApplyDict(nopts)
            # Name of output file.
            fout = os.path.join(frun, 'fun3d.%02i.nml' % j)
            # Write the input file.
            self.Namelist.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
        
        
        
        
    # Write run control options to JSON file
    def WriteCaseJSON(self, i):
        """Write JSON file with run control and related settings for case *i*
        
        :Call:
            >>> fun3d.WriteCaseJSON(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2015-10-19
            ``@ddalle``: First version
        """
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            # Go back and quit.
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Write folder.
        f = open('case.json', 'w')
        # Dump the flowCart settings.
        json.dump(self.opts['flowCart'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
# class Fun3d

