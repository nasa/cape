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
from . import dataBook
from . import manage
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
        
        # Read the Config.xml file
        self.ReadConfig()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyOver.Overflow(nCase=%i)>" % (
            self.x.nCase)
        
    # Function to read the databook.
    def ReadDataBook(self):
        """Read the current data book
        
        :Call:
            >>> oflow.ReadDataBook()
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
        :Versions:
            * 2016-02-17 ``@ddalle``: First version
        """
        # Test for an existing data book.
        try:
            self.DataBook
            return
        except AttributeError:
            pass
        # Go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the data book.
        self.DataBook = dataBook.DataBook(self.x, self.opts)
        # Return to original folder.
        os.chdir(fpwd)
        
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
        # File name
        fnml = self.opts.get_OverNamelist(j)
        # Check for the file.
        if not os.path.isfile(fnml):
            # Do nothing
            nml = None
        else:
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
        # ---------------
        # Config.xml prep
        # ---------------
        # Process configuration
        self.PrepareConfig(i)
        # Write the file if possible
        self.WriteConfig(i)
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
            if os.path.isfile(f0): os.symlink(f0, f1)
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
    def GetCPUTime(self, i, running=False):
        """Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = oflow.GetCPUTime(i, running=False)
        :Inputs:
            *oflow*: :class:`pyFun.fun3d.Fun3d`
                OVERFLOW control interface
            *i*: :class:`int`
                Case index
            *running*: ``True`` | {``False``}
                Whether or not the case is running
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
            * 2016-08-31 ``@ddalle``: Added start times
        """
        # File names
        fname = 'pyover_time.dat'
        fstrt = 'pyover_start.dat'
        # Call the general function using hard-coded file name
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)
    
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
        # Prepare the Config.xml translations and rotations
        self.PrepareConfig(i)
        # Write the over.namelist file(s).
        self.PrepareNamelist(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the configuration file
        self.WriteConfig(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location
        os.chdir(fpwd)
        
        
    # Function to prepare "input.cntl" files
    def PrepareNamelist(self, i, nPhase=None):
        """
        Write :file:`over.namelist` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        The optional input *nPhase* can be used to right additional phases that
        are not part of the default *PhaseSequence*, which can be useful when
        only a subset of cases in the run matrix will require additional
        phases.
        
        :Call:
            >>> oflow.PrepareNamelist(i, nPhase=None)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Instance of OVERFLOW control class
            *i*: :class:`int`
                Run index
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
            * 2016-12-13 ``@ddalle``: Added second input variable
        """
        # Phase number
        if nPhase is None:
            nPhase = self.opt.get_nSeq()
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
        M = x.GetMach(i)
        if M  is not None: self.Namelist.SetMach(M)
        # Angle of attack
        a = x.GetAlpha(i)
        if a  is not None: self.Namelist.SetAlpha(a)
        # Sideslip angle
        b = x.GetBeta(i)
        if b  is not None: self.Namelist.SetBeta(b)
        # Reynolds number
        Re = x.GetReynoldsNumber(i)
        if Re is not None: self.Namelist.SetReynoldsNumber(Re)
        # Temperature
        T = x.GetTemperature(i)
        if T  is not None: self.Namelist.SetTemperature(T)
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): return
        
        # Set the surface BCs
        for k in self.x.GetKeysByType('SurfBC'):
            # Apply the appropriate methods
            self.SetSurfBC(k, i)
        # Set the surface BCs that use thrust as input
        for k in self.x.GetKeysByType('SurfCT'):
            # Apply the appropriate methods
            self.SetSurfBC(k, i, CT=True)
            
        # Loop through input sequence
        for j in range(nPhase):
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
        
    # Function to apply settings from a specific JSON file
    def ApplyNamelistSettings(self, nPhase=None, **kw):
        """Apply settings from *oflow.opts* to a set of cases
        
        This rewrites each run namelist file and the :file:`case.json` file in
        the specified directories.  It can also be used to 
        
        :Call:
            >>> oflow.ApplyNamelistSettings(cons=[], **kw)
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Overflow control interface
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints
        :Versions:
            * 2014-12-11 ``@ddalle``: First version
        """
        # Apply filter.
        I = self.x.GetIndices(**kw)
        # Current phase number
        nSeq = self.opts.get_nSeq()
        # Phase number
        if nPhase is None:
            # Use the current phase number
            nPhase = nSeq
        else:
            # Make sure it's an integer
            nPhase = int(nPhase)
        # Loop through cases.
        for i in I:
            # Read the case json

            # Check for nPhase greater than that in *PhaseSequence*
            if nPhase > nSeq:
                # Append the new phase(s)
                for j in range(nSeq, nPhase):
                    self.opts["RunControl"]["PhaseSequence"].append(j)
            # Write the JSON file.
            self.WriteCaseJSON(i)
            # Write namelist
            self.PrepareNamelist(i, nPhase)
        
    # Write configuration file
    def WriteConfig(self, i, fname='Config.xml'):
        """Write configuration file
        
        :Call:
            >>> oflow.WriteConfig(i, fname='Config.xml')
        :Inputs:
            *oflow*: :class:`pyOver.overflow.Overflow`
                Overflow control interface
            *i*: :class:`int`
                Case index
            *fname*: {``'Config.xml'``} | :class:`str`
                Name of file to write within run folder
        :Versions:
            * 2016-08-24 ``@ddalle``: First version
            * 2016-08-26 ``@ddalle``: 
        """
        # Safely go to root.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check for existence.
        if not os.path.isdir(frun):
            # Go back and quit
            os.chdir(fpwd)
            return
        # Go to the run folder
        os.chdir(frun)
        # Write the file if possible
        try:
            # Test if the Config is present
            self.config
        except AttributeError:
            # No config; ok not to write
            pass
        else:
            # Write the file if the above does not fail
            if self.config is not None:
                self.config.Write(fname)
        # Return to original location
        os.chdir(fpwd)
        
    # Prepare surface BC
    def SetSurfBC(self, key, i, CT=False):
        """Set a surface BC for one key using *IBTYP* 153
        
        :Call:
            >>> ofl.SetSurfBC(key, i, CT=False)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*, *T0* directly)
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Get list of grids
        grids = self.x.GetSurfBC_Grids(i, key)
        # Get the current namelist
        nml = self.Namelist
        # Case folder
        frun = os.path.join(self.RootDir, self.x.GetFullFolderNames(i))
        # Safely go to the folder
        fpwd = os.getcwd()
        if os.path.isdir(frun): os.chdir(frun)
        # Loop through the grids
        for grid in grids:
            # Get component
            comp = self.x.GetSurfBC_CompID(i, key, comp=grid)
            # BC index
            bci = self.x.GetSurfBC_BCIndex(i, key, comp=grid)
            # Boundary conditions
            if CT:
                # Get *p0*, *T0* from thrust
                p0, T0 = self.GetSurfCTState(key, i, grid)
                # Key type
                typ = 'SurfCT'
            else:
                # Use *p0* and *T0* directly as inputs
                p0, T0 = self.GetSurfBCState(key, i, grid)
                # Key type
                typ = 'SurfBC'
            # Species
            Y = self.x.GetSurfBC_Species(i, key, comp=grid)
            # Other parameters
            BCPAR1 = self.x.GetSurfBC_Param(i, key, 'BCPAR1', 
                comp=grid, typ=typ, vdef=1)
            BCPAR2 = self.x.GetSurfBC_Param(i, key, 'BCPAR2',
                comp=grid, typ=typ, vdef=500)
            # File name
            fname = 'SurfBC-%s-%s.dat' % (comp, grid)
            # Open the file
            f = open(fname, 'w')
            # Write the state
            f.write("%.12f %.12f\n" % (p0, T0))
            # Write the species info
            if Y is not None and len(Y) > 1:
                f.write(" ".join([str(y) for y in Y]))
                f.write("\n")
            # Write the component name
            f.write("%s\n" % comp)
            # Write the time history
            # Writing just 1 causes Overflow to ignore, but there still needs
            # to be one row of time value and thrust value (as a percentage)
            f.write("1\n")
            f.write("%.12f %.12f\n" % (1.0, 1.0))
            f.close()
            # Get the BC number parameter
            IBTYP = nml.GetKeyFromGrid(grid, 'BCINP', 'IBTYP')
            # Check for list
            if type(IBTYP).__name__ in ['list', 'ndarray']:
                # Check for at least *bci* columns
                if bci > len(IBTYP):
                    raise ValueError(
                        ("While specifying IBTYP for key '%s':\n" % key) +
                        ("Received column index %s for grid '%s' " % (bci, grid)) +
                        ("but BCINP/IBTYP namelist has %s columns" % len(IBTYP)))
                # Set IBTYP to 153
                IBTYP[bci-1] = 153
            else:
                # Make sure *bci* is 1
                if bci != 1:
                    raise ValueError(
                        ("While specifying IBTYP for key '%s':\n" % key) +
                        ("Received column index %s for grid '%s' " % (bci, grid)) +
                        ("but BCINP/IBTYP namelist has 1 column" % len(IBTYP)))
                # Set IBTYP to 153
                IBTYP = 153
            # Reset IBTYP
            nml.SetKeyForGrid(grid, 'BCINP', 'IBTYP', IBTYP)
            # Set the parameters in the namelist
            nml.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', BCPAR1, i=bci)
            nml.SetKeyForGrid(grid, 'BCINP', 'BCPAR2', BCPAR2, i=bci)
            nml.SetKeyForGrid(grid, 'BCINP', 'BCFILE', fname,  i=bci)
        # Return to original location
        os.chdir(fpwd)            
            
    # Get surface BC inputs
    def GetSurfBCState(self, key, i, grid=None):
        """Get stagnation pressure and temperature ratios
        :Call:
            >>> p0, T0 = ofl.GetSurfBC(key, i, grid=None)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *grid*: {``None``} | :class:`str`
                Name of grid for which to extract settings
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC total pressure to freestream total pressure
            *T0*: :class:`float`
                Ratio of BC total temperature to freestream total temperature
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Get the inputs
        p0 = self.x.GetSurfBC_TotalPressure(i, key, comp=grid)
        T0 = self.x.GetSurfBC_TotalTemperature(i, key, comp=grid)
        # Calibration
        bp = self.x.GetSurfBC_PressureOffset(i, key, comp=grid)
        ap = self.x.GetSurfBC_PressureCalibration(i, key, comp=grid)
        bT = self.x.GetSurfBC_TemperatureOffset(i, key, comp=grid)
        aT = self.x.GetSurfBC_TemperatureCalibration(i, key, comp=grid)
        # Reference pressure/tomp
        p0inf = self.x.GetSurfBC_RefPressure(i, key, comp=grid)
        T0inf = self.x.GetSurfBC_RefTemperature(i, key, comp=grid)
        # Output
        return (ap*p0+bp)/p0inf, (aT*T0+bT)/T0inf
        
    # Get surface *CT* state inputs
    def GetSurfCTState(self, key, i, grid=None):
        """Get stagnation pressure and temperature ratios for *SurfCT* key
        
        :Call:
            >>> p0, T0 = ofl.GetSurfCTState(key, i, grid=None)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *grid*: {``None``} | :class:`str`
                Name of grid for which to extract settings
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC total pressure to freestream total pressure
            *T0*: :class:`float`
                Ratio of BC total temperature to freestream total temperature
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Get the trhust value
        CT = self.x.GetSurfCT_Thrust(i, key, comp=grid)
        # Get the exit parameters
        M2 = self.x,GetSurfCT_ExitMach(i, key, comp=grid)
        A2 = self.x,GetSurfCT_ExitArea(i, key, comp=grid)
        # Ratio of specific heats
        gam = self.x.GetSurfCT_Gamma(i, key, comp=grid)
        # Derivative gas constants
        g2 = 0.5 * (gam-1)
        g3 = gam / (gam-1)
        # Get reference dynamice pressure
        qref = self.GetSurfCT_RefDynamicPressure(i, key, comp=grid)
        # Get reference area
        Aref = self.GetSurfCT_RefArea(key, i)
        # Calculate total pressure
        p0 = CT*qref*aref/A2 *  (1+g2*M2*M2)**g3 / (1+gam*M2*M2)
        # Temperature inputs
        T0 = self.x.GetSurfCT_TotalTemperature(i, key, comp=grid)
        # Calibration
        ap = self.x.GetSurfCT_PressureCalibration(i, key, comp=grid)
        bp = self.x.GetSurfCT_PressureOffset(i, key, comp=grid)
        aT = self.x.GetSurfCT_TemperatureCalibration(i, key, comp=grid)
        bT = self.x.GetSurfCT_TemperatureOffset(i, key, comp=grid)
        # Reference values
        p0inf = self.x.GetSurfCT_TotalPressure(i, key, comp=grid)
        T0inf = self.x.GetSurfCT_TotalTemperature(i, key, comp=grid)
        # Output
        return (ap*p0+bp)/p0inf, (aT*T0+bT)/T0inf
        
    # Write run control options to JSON file
    def WriteCaseJSON(self, i):
        """Write JSON file with run control and related settings for case *i*
        
        :Call:
            >>> ofl.WriteCaseJSON(i)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
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
        # Dump the Overflow and other run settings.
        json.dump(self.opts['RunControl'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
    # Read run control options from case JSON file
    def ReadCaseJSON(self, i):
        """Read ``case.json`` file from case *i* if possible
        
        :Call:
            >>> rc = ofl.ReadCaseJSON(i)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`pyOver.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
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
        # Check for file
        if not os.path.isfile('case.json'):
            # Nothing to read
            rc = None
        else:
            # Read the file
            rc = case.ReadCaseJSON()
        # Return to original location
        os.chdir(fpwd)
        # Output
        return rc
        
    # Read a namelist from a case folder
    def ReadCaseNamelist(self, i, rc=None, j=None):
        """Read namelist from case *i*, phase *j* if possible
        
        :Call:
            >>> nml = ofl.ReadCaseNamelist(i, rc=None, j=None)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Run index
            *rc*: ``None`` | :class:`pyOver.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *nml*: ``None`` | :class:`pyOver.overNamelist.OverNamelist`
                Namelist interface is possible
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
        """
        # Read the *rc* if necessary
        if rc is None:
            rc = self.ReadCaseJSON(i)
        # If still None, exit
        if rc is None: return
        # Get phase number
        if j is None:
            j = rc.get_PhaseSequence(-1)
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
        # Read the namelist
        nml = case.GetNamelist(rc, j)
        # Return to original location
        os.chdir(fpwd)
        # Output
        return nml
        
    # Extend a case
    def ExtendCase(self, i, n=1, j=None, imax=None):
        """Add *NSTEPS* iterations to case *i* using the last phase's namelist
        
        :Call:
            >>> ofl.ExtendCase(i, n=1, j=None, imax=None)
        :Inputs:
            *ofl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control class
            *i*: :class:`int`
                Run index
            *n*: {``1``} | positive :class:`int`
                Add *n* times *NSTEPS* to the total iteration count
            *j*: {``None``} | nonnegative :class:`int`
                Apply to phase *j*, by default use the last phase
            *imax*: {``None``} | nonnegative :class:`int`
                Use *imax* as the maximum iteration count
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
        """
        # Read the ``case.json`` file
        rc = self.ReadCaseJSON(i)
        # Exit if none
        if rc is None: return
        # Read the namelist
        nml = self.ReadCaseNamelist(i, rc, j=j)
        # Exit if that's None
        if nml is None: return
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Safely go to the run directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        os.chdir(frun)
        # Get the phase number
        j = rc.get_PhaseSequence(-1)
        # Get the number of steps
        NSTEPS = nml.GetKeyFromGroupName("GLOBAL", "NSTEPS")
        # Get the current cutoff for phase *j*
        N = rc.get_PhaseIters(j)
        # Determine output number of steps
        if imax is None:
            # Unlimited by input; add one or more nominal runs
            N1 = N + n*NSTEPS
        else:
            # Add nominal runs but not beyond *imax*
            N1 = min(int(imax), int(N + n*NSTEPS))
        # Reset the number of steps
        rc.set_PhaseIters(N1, j)
        # Status update
        print("  Phase %i: %s --> %s" % (j, N, N1))
        # Write folder.
        f = open('case.json', 'w')
        # Dump the Overflow and other run settings.
        json.dump(rc, f, indent=1)
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
        
    
    # Individual case archive function
    def ArchivePWD(self):
        """Archive a single case in the current folder ($PWD)
        
        :Call:
            >>> oflow.ArchivePWD()
        :Inputs:
            *cntl*: :class:`pyOver.overflow.Overflow`
                Instance of pyOver control interface
        :Versions:
            * 2016-12-09 ``@ddalle``: First version
        """
        # Archive using the local module
        manage.ArchiveFolder(self.opts)
        
# class Overflow

