"""
OVERFLOW control module: :mod:`pyOver.overflow`
===============================================

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
from overNamelist import OverNamelist

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
PyOverFolder = os.path.split(_fname)[0]
    
# Class to read input files
class Overflow(Cntl):
    """
    Class for handling global options and setup for OVERFLOW.
    
    This class is intended to handle all settings used to describe a group
    of OVERFLOW cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha sweep
    for each single control variable setting.
    
    The settings are read from a JSON file, which is robust and simple to
    read, but has the disadvantage that there is no support for comments.
    Hopefully the various names are descriptive enough not to require
    explanation.
    
    Defaults are read from the file ``$CAPE/settings/pyOver.default.json``.
    
        >>> oflow = pyFlow.Overflow(fname="pyFlow.json")
        >>> oflow = pyOver.Overflow(fname="pyOver.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyFun input file
    :Outputs:
        *oflow*: :class:`pyFun.fun3d.Fun3d`
            Instance of the pyFun control class
    :Data members:
        *oflow.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *oflow.x*: :class:`pyOver.trajectory.Trajectory`
            Values and definitions for variables in the run matrix
        *oflow.Namelist*: :class:`pyOver.overNamelist.OverNamelist`
            Interface to ``over.namelist`` OVERFLOW input file
        *oflow.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: Started
        * 2016-02-02 ``@ddalle``: First version
    """ 
    # Initialization method
    def __init__(self, fname="pyOver.json"):
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
        
        # Read the namelist
        self.ReadNamelist()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyOver.Overflow(nCase=%i)>" % (
            self.x.nCase)
        
    # Read namelist
    def ReadNamelist(self, j=0, q=True):
        """Read the OVERFLOW namelist template
        
        :Call:
            >>> oflow.ReadNamelist(j=0, q=True)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Change to root safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file
        nml = OverNamelist(self.opts.get_OverNamelist(j))
        # Save it
        if q:
            # Read to main slot for modification
            self.Namelist = nml
        else:
            # Template for reading original parameters
            self.Namelist0 = nml
        # Go back to original location
        os.chdir(fpwd)
        
    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        """Get a namelist variable's value
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> val = oflow.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *sec*: :class:`str`
                Name of namelist section/group
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the namelist value.
        nval = self.Namelist.GetKeyFromGroupName(sec, key)
        # Check for options value.
        if nval is None:
            # No namelist file value
            return self.opts.get_namelist_var(sec, key, j)
        elif 'Overflow' not in self.opts:
            # No namelist in options
            return nval
        elif sec not in self.opts['Overflow']:
            # No corresponding options section
            return nval
        elif key not in self.opts['Overflow'][sec]:
            # Value not specified in the options namelist section
            return nval
        else:
            # Default to the options
            return self.opts_get_namelist_var(sec, key, i)
        
    # Get the project rootname
    def GetPrefix(self, j=0):
        """Get the project root name or OVERFLOW file prefix
        
        :Call:
            >>> name = oflow.GetPrefix(j=0)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Return the value from options
        return self.opts.get_Prefix(j)
        
    # Get the project rootname
    def GetConfig(self, i):
        """Get the configuration (if any) for case *i*
        
        If there is no *config* or similar run matrix variable, return the name
        of the group folder
        
        :Call:
            >>> config = oflow.GetConfig(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Case index
        :Outputs:
            *config*: :class:`str`
                Case configuration
        :Versions:
            * 2016-02-02 ``@ddalle``: First version
        """
        # Check for configuration variable(s)
        xcfg = self.x.GetKeysByType("Config")
        # If there's no match, return group folder
        if len(xcfg) == 0:
            # Return folder name
            return self.x.GetGroupFolderNames(i)
        # Initialize output
        config = ""
        # Loop through config variables (weird if more than one...)
        for k in xcfg:
            config += getattr(self.x,k)[i]
        # Output
        return config
        
    # Function to get configuration directory
    def GetConfigDir(self, i):
        """Return absolute path to configuration folder
        
        :Call:
            >>> fcfg = oflow.GetConfigDir(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Case index
        :Outputs:
            *fcfg*: :class:`str`
                Full path to configuration folder
        :Versions:
            * 2016-02-02 ``@ddalle``: First version
        """
        # Configuration of this case
        config = self.GetConfig(i)
        # Configuration folder
        fcfg = self.opts.get_ConfigDir(config)
        # Check if it begins with a slash.
        if os.path.isabs(fcfg):
            # Return as absolute path
            return fcfg
        else:
            # Append the root directory
            return os.path.join(self.RootDir, fcfg)

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
        # Read value
        n = case.GetCurrentIter()
        # Default to zero.
        if n is None:
            return 0
        else:
            return n
        
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
        
        
    # Get list of raw file names
    def GetMeshFileNames(self, i=0):
        """Return the list of mesh files
        
        :Call:
            >>> fname = fun3d.GetMeshFileNames()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *fname*: :class:`list` (:class:`str`)
                List of file names read from root directory
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get config
        config = self.GetConfig(i)
        # Get the file names from *opts*
        fname = self.opts.get_MeshFiles(config)
        # Remove folders
        fname = [os.path.split(f)[-1] for f in fname]
        # Output
        return fname
        
    # Function to check if the mesh for case *i* is prepared
    def CheckMesh(self, i):
        """Check if the mesh for case *i* is prepared
        
        :Call:
            >>> q = oflow.CheckMesh(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of OVERFLOW run control class
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Check input
        if not type(i).__name__.startswith("int"):
            raise TypeError("Case index must be an integer")
        # Get the case folder name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize with a "pass" location
        q = True
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check for the group folder.
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return False
        # Extract options
        opts = self.opts
        # Enter the case folder.
        os.chdir(frun)
        # Get list of mesh file names.
        fmesh = self.GetMeshFileNames(i)
        # Check for presence
        for f in fmesh:
            # Check for the file
            q = q and os.path.isfile(f)
        # Return to original folder.
        os.chdir(fpwd)
        # Output
        return q
        
    # Prepare the mesh for case *i* (if necessary)
    def PrepareMesh(self, i):
        """Prepare the mesh for case *i* if necessary
        
        :Call:
            >>> oflow.PrepareMesh(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # ---------
        # Case info
        # ---------
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group.
        fgrp = self.x.GetGroupFolderNames(i)
        # Check the mesh.
        if self.CheckMesh(i):
            return None
        # ------------------
        # Folder preparation
        # ------------------
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the group folder and make it if necessary.
        if not os.path.isdir(fgrp):
            self.mkdir(fgrp)
        # Check if the fun folder exists.
        if not os.path.isdir(frun):
            self.mkdir(frun)
        # Status update
        print("  Case name: '%s' (index %i)" % (frun,i))
        # Enter the case folder.
        os.chdir(frun)
        # ----------
        # Copy files
        # ----------
        # Configuration of this case
        config = self.GetConfig(i)
        # Get the configuration folder
        fcfg = self.GetConfigDir(i)
        # Get the names of the raw input files and target files
        fmsh = self.opts.get_MeshCopyFiles(config)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Replace 'x.save' -> 'x.restart'
            f1 = f1.replace('save', 'restart')
            # Remove the file if necessary
            if os.path.islink(f1): os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1): continue
            # Link the file.
            if os.path.isfile(f0):
                shutil.copy(f0, f1)
        # Get the names of input files to copy
        fmsh = self.opts.get_MeshLinkFiles(config)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Replace 'x.save' -> 'x.restart'
            f1 = f1.replace('save', 'restart')
            # Remove the file if necessary
            if os.path.islink(f1): os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1): continue
            # Link the file.
            if os.path.isfile(f0):
                os.symlink(f0, f1)
        # -------
        # Cleanup
        # -------
        # Return to original folder
        os.chdir(fpwd)
        
        
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self):
        """Check if the current folder has the necessary files to run
        
        :Call:
            >>> q = fun3d.CheckNone()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Input file.
        finp = '%s.01.inp' % self.GetPrefix()
        if not os.path.isfile(finp): return True
        # Settings file.
        if not os.path.isfile('case.json'): return True
        # Get mesh file names
        fmsh = self.GetMeshFileNames()
        # Check for them.
        for fi in fmsh:
            # Check for modified file name: 'save' -> 'restart'
            fo = fi.replace('save', 'restart')
            # Check for the file
            if not os.path.isfile(fo): return True
        # Apparently no issues.
        return False
            
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i):
        """Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = oflow.GetCPUTime(i)
        :Inputs:
            *oflow*: :class:`pyFun.fun3d.Fun3d`
                OVERFLOW control interface
            *i*: :class:`int`
                Case index
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
        """
        # Call the general function using hard-coded file name
        return self.GetCPUTimeFromFile(i, fname='pyover_time.dat')

    
    # Prepare a case.
    def PrepareCase(self, i):
        """Prepare a case for running if it is not already prepared
        
        :Call:
            >>> fun3d.PrepareCase(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if already prepared.
        if n is not None: return
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Enter the run directory.
        if not os.path.isdir(frun): self.mkdir(frun)
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Get function for setting boundary conditions, etc.
        keys = self.x.GetKeysByType('CaseFunction')
        # Get the list of functions.
        funcs = [self.x.defns[key]['Function'] for key in keys]
        # Reread namelist
        self.ReadNamelist()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Apply it.
            exec("%s(self,%s,i=%i)" % (func, getattr(self.x,key)[i], i))
        # Write the over.namelist file(s).
        self.PrepareNamelist(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location
        os.chdir(fpwd)
        
        
    # Function to prepare "input.cntl" files
    def PrepareNamelist(self, i):
        """
        Write :file:`over.namelist` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> oflow.PrepareNamelist(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of OVERFLOW control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
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
        # Check for Reynolds number
        if 'Re' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('Re')[0]
            # Set the value.
            self.Namelist.SetReynoldsNumber(getattr(x,k)[i])
        # Check for temperature
        if 'T' in KeyTypes:
            # Find the key.
            k = x.GetKeysByType('T')[0]
            # Set the value.
            self.Namelist.SetTemperature(getattr(x,k)[i])
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Loop through input sequence
        for j in range(self.opts.get_nSeq()):
            # Set the "restart_read" property appropriately
            # This setting is overridden by *nopts* if appropriate
            if j == 0:
                # First run sequence; not restart
                self.Namelist.SetRestart(False)
            else:
                # Later sequence; restart
                self.Namelist.SetRestart(True)
            # Set number of iterations
            self.Namelist.SetnIter(self.opts.get_nIter(j))
            # Get the reduced namelist for sequence *j*
            nopts = self.opts.select_namelist(j)
            # Apply them to this namelist
            self.Namelist.ApplyDict(nopts)
            # Get options to apply to all grids
            oall = self.opts.get_ALL(j)
            # Apply those options
            self.Namelist.ApplyDictToALL(oall)
            # Loop through other custom grid systems
            for grdnam in self.opts.get('Grids',{}):
                # Skip for key 'ALL'
                if grdnam == 'ALL': continue
                # Get options for this grid
                ogrd = self.opts.get_GridByName(grdnam, j)
                # Apply the options
                self.Namelist.ApplyDictToGrid(grdnam, ogrd)
            # Name of output file.
            fout = os.path.join(frun, '%s.%02i.inp' % (self.GetPrefix(j), j+1))
            # Write the input file.
            self.Namelist.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
        
    # Set up a namelist config
    def PrepareNamelistConfig(self):
        """Write the lines for the force/moment output in a namelist file
        
        :Call:
            >>> fun3d.PrepareNamelistConfig()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Get the components
        comps = self.opts.get_ConfigComponents()
        # Number
        n = len(comps)
        # Quit if nothing to do
        if n == 0: return
        # Extract namelist
        nml = self.Namelist
        # Loop through specified components.
        for k in range(n,0,-1):
            # Get component.
            comp = comps[k-1]
            # Get input definitions.
            inp = self.opts.get_ConfigInput(comp)
            # Set input definitions.
            if inp is not None:
                nml.SetVar('component_parameters', 'component_input', inp, k)
            # Reference area
            if 'RefArea' in self.opts['Config']:
                # Get reference area.
                RefA = self.opts.get_RefArea(comp)
                # Set it
                nml.SetVar('component_parameters', 'component_sref', RefA, k)
            # Moment reference center
            if 'RefPoint' in self.opts['Config']:
                # Get MRP
                RefP = self.opts.get_RefPoint(comp)
                # Set the x- and y-coordinates
                nml.SetVar('component_parameters', 'component_xmc', RefP[0], k)
                nml.SetVar('component_parameters', 'component_ymc', RefP[1], k)
                # Check for z-coordinate
                if len(RefP) > 2:
                    nml.SetVar(
                        'component_parameters', 'component_zmc', RefP[2], k)
            # Reference length
            if 'RefLength' in self.opts['Config']:
                # Get reference length
                RefL = self.opts.get_RefLength(comp)
                # Set both reference lengths
                nml.SetVar('component_parameters', 'component_cref', RefL, k)
                nml.SetVar('component_parameters', 'component_bref', RefL, k)
            # Set the component name
            nml.SetVar('component_parameters', 'component_name', comp, k)
            # Tell FUN3D to determine the number of components on its own.
            nml.SetVar('component_parameters', 'component_count', -1, k)
        # Set the number of components
        nml.SetVar('component_parameters', 'number_of_components', n)
        
        
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
            * 2015-10-19 ``@ddalle``: First version
            * 2016-02-01 ``@ddalle``: Copied from :mod:`pyFun`
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
        json.dump(self.opts['RunControl'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
    # Write the PBS script.
    def WritePBS(self, i):
        """Write the PBS script(s) for a given case
        
        :Call:
            >>> oflow.WritePBS(i)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-10-19 ``@ddalle``: First version
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to the root directory.
        os.chdir(self.RootDir)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go to the folder.
        os.chdir(frun)
        # Determine number of unique PBS scripts.
        if self.opts.get_nPBS() > 1:
            # If more than one, use unique PBS script for each run.
            nPBS = self.opts.get_nSeq()
        else:
            # Otherwise use a single PBS script.
            nPBS = 1
        
        # Loop through the runs.
        for j in range(nPBS):
            # PBS script name.
            if nPBS > 1:
                # Put PBS number in file name.
                fpbs = 'run_overflow.%02i.pbs' % (j+1)
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_overflow.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)
            
            # Initialize options to `run_FUN3D.py`
            flgs = ''

            # Simply call the advanced interface.
            f.write('\n# Call the OVERFLOW interface.\n')
            f.write('run_overflow.py' + flgs)
            
            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)
        
    # Call the correct :mod:`case` module to start a case
    def CaseStartCase(self):
        """Start a case by either submitting it or running it
        
        This function relies on :mod:`pyCart.case`, and so it is customized for
        the Cart3D solver only in that it calles the correct *case* module.
        
        :Call:
            >>> pbs = oflow.CaseStartCase()
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.StartCase()
        
        
# class Overflow

