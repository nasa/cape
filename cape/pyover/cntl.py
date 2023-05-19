# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyover.cntl`: OVERFLOW control module 
==================================================

This module provides tools to quickly setup basic or complex OVERFLOW run matrices
and serve as an executive for pre-processing, running, post-processing, and
managing the solutions. A collection of cases combined into a run matrix can be
loaded using the following commands.

    .. code-block:: pycon
    
        >>> import cape.pyover.fun3d
        >>> cntl = cape.pyover.cntl.Cntl("pyOver.json")
        >>> cntl
        <cape.pyover.Cntl(nCase=907)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'
        
        
An instance of this :class:`cape.pyover.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.Namelist``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pyover.options.Options`
    *cntl.DataBook*        :class:`cape.pyover.dataBook.DataBook`
    *cntl.Namelist*        :class:`cape.pyover.namelist.Namelist`
    ====================   =============================================

Finally, the :class:`Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE class are
also available here.

"""

# Standard library
import os
import json
import shutil
import subprocess as sp
from datetime import datetime

# Third-party
import numpy as np

# Local imports
from . import options
from . import case
from . import dataBook
from . import manage
from . import report
from .. import cntl as capecntl
from .. import convert
from .overNamelist import OverNamelist
from ..runmatrix import RunMatrix


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyOverFolder = os.path.split(_fname)[0]
    
# Class to read input files
class Cntl(capecntl.Cntl):
    r"""Class for handling global options and setup for OVERFLOW

    This class is intended to handle all settings used to describe a
    group of OVERFLOW cases. For situations where it is not
    sufficiently customized, it can be used partially, e.g., to set up a
    Mach/alpha sweep for each single control variable setting.

    The settings are read from a JSON file, which is robust and simple
    to read, but has the disadvantage that there is no support for
    comments. Hopefully the various names are descriptive enough not to
    require explanation.
    
    :Call:
        >>> cntl = cape.pyover.Cntl(fname="pyOver.json")
    :Inputs:
        *fname*: :class:`str`
            Name of cape.pyover input file
    :Outputs:
        *cntl*: :class:`cape.pyfun.cntl.Cntl`
            Instance of the cape.pyover control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`cape.pyover.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.Namelist*: :class:`cape.pyover.overNamelist.OverNamelist`
            Interface to ``over.namelist`` OVERFLOW input file
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: Started
        * 2016-02-02 ``@ddalle``: Version 1.0
    """
  # =================
  # Class Attributes
  # =================
  # <
    # Case module
    _case_mod = case
  # >

  # =======
  # Config
  # =======
  # <
    # Initialization method
    def __init__(self, fname="pyOver.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        # Force default
        if fname is None:
            fname = "pyOver.json"
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No cape.pyover control file '%s' found" % fname)
        
        # Read settings
        self.opts = options.Options(fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.modules = {}
        self.ImportModules()
        
        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}
        
        # Read the namelist
        self.ReadNamelist()
        
        # Read the Config.xml file
        self.ReadConfig()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
        # Run any initialization functions
        self.InitFunction()
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<cape.pyover.Cntl(nCase=%i)>" % (
            self.x.nCase)
  # >
  
  # =======================
  # Command-Line Interface
  # =======================
  # <
    # Baseline function
    def cli(self, *a, **kw):
        r"""Command-line interface
        
        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of control class containing relevant parameters
            *kw*: :class:`dict` (``True`` | ``False`` | :class:`str`)
                Unprocessed keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: Content from ``bin/`` executables
        """
        # Preprocess command-line inputs
        a, kw = self.cli_preprocess(*a, **kw)
        # Call the common interface
        cmd = self.cli_cape(*a, **kw)
        # Test for a command
        if cmd is not None:
            return
        # Otherwise fall back to code-specific commands
        if kw.get('stop'):
            # Update point sensor data book
            self.StopCases(n=kw['stop'], **kw)
        else:
            # Submit the jobs
            self.SubmitJobs(**kw)

    # Function to apply namelist settings to a case
    def ApplyCase(self, i, nPhase=None, **kw):
        r"""Apply settings from *cntl.opts* to a set of cases
        
        This rewrites each run namelist file and the ``case.json`` file
        in the specified directories. 
        
        :Call:
            >>> cntl.ApplyCase(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Overflow control interface
            *i*: :class:`int`
                Case number
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2014-12-13 ``@ddalle``: Version 1.0
        """
        # Ignore cases marked PASS
        if self.x.PASS[i]: return
        # Case function
        self.CaseFunction(i)
        # Read ``case.json``.
        rc = self.ReadCaseJSON(i)
        # Get present options
        rco = self.opts["RunControl"]
        # Exit if none
        if rc is None: return
        # Get the number of phases in ``case.json``
        nSeqC = rc.get_nSeq()
        # Get number of phases from present options
        nSeqO = self.opts.get_nSeq()
        # Check for input
        if nPhase is None:
            # Default: inherit from pyOver.json
            nPhase = nSeqO
        else:
            # Use maximum
            nPhase = max(nSeqC, int(nPhase))
        # Present number of iterations
        nIter = rc.get_PhaseIters(nSeqC)
        # Loop through the additional phases
        for j in range(nSeqC, nPhase):
            # Append the new phase
            rc["PhaseSequence"].append(j)
            # Get iterations for this phase
            if j > nSeqO:
                # Get nIter for phase *j*
                nj = self.opts.get_namelist_var('GLOBAL', 'NSTEPS', j)
            else:
                # Use the phase break marker from master JSON file
                nj = self.opts.get_PhaseIters(j) - nIter
            # Get iterations for this phase
            # Status update
            print("  Adding phase %s (to %s iterations)" % (j, nIter+nj))
            # Set the iteration count
            nIter += nj
            rc.set_PhaseIters(nIter, j)
            # Copy other sections
            for k in rco:
                # Don't copy phase and iterations
                if k in ["PhaseIters", "PhaseSequence"]: continue
                # Otherwise, overwrite
                rc[k] = rco[k]
        # Write it
        self.WriteCaseJSON(i, rc=rc)
        # Write the conditions to a simple JSON file
        self.WriteConditionsJSON(i)
        # Reread source namelist template
        self.ReadNamelist()
        # Rewriting phases
        print("  Writing input namelists 1 to %s" % (nPhase))
        self.PrepareNamelist(i, nPhase)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 1 to %s" % (nPBS))
        self.WritePBS(i)

    # Extend a case
    def ExtendCase(self, i, n=1, j=None, imax=None):
        r"""Run the final phase of case *i* again
        
        :Call:
            >>> cntl.ExtendCase(i, n=1, j=None, imax=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Run index
            *n*: {``1``} | positive :class:`int`
                Add *n* times *NSTEPS* to the total iteration count
            *j*: {``None``} | nonnegative :class:`int`
                Apply to phase *j*, by default use the last phase
            *imax*: {``None``} | nonnegative :class:`int`
                Use *imax* as the maximum iteration count
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
        """
        # Ignore cases marked PASS
        if self.x.PASS[i]: return
        # Read the ``case.json`` file
        rc = self.ReadCaseJSON(i)
        # Exit if none
        if rc is None: return
        # Read the namelist
        nml = self.ReadCaseNamelist(i, rc, j=j)
        # Exit if that's None
        if nml is None: return
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
        # Write new options
        self.WriteCaseJSON(i, rc=rc)
  # >

  # =================
  # Primary prep
  # =================
  # <
    # Prepare a case.
    def PrepareCase(self, i):
        r"""Prepare a case for running if it is not already prepared
        
        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: Version 1.0
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if already prepared.
        if n is not None:
            return
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Enter the run directory.
        if not os.path.isdir(frun):
            self.mkdir(frun)
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
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixCaseFunction")
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
        r""" Write ``over.namelist`` for run case *i*
        
        The optional input *nPhase* can be used to right additional
        phases that are not part of the default *PhaseSequence*, which
        can be useful when only a subset of cases in the run matrix will
        require additional phases.
        
        :Call:
            >>> cntl.PrepareNamelist(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of OVERFLOW control class
            *i*: :class:`int`
                Run index
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
            * 2016-12-13 ``@ddalle``: Version 1.1; add second input
            * 2022-01-25 ``@ddalle``: Version 1.2; reread nml each call
        """
        # Reread namelist
        self.ReadNamelist()
        # Phase number
        if nPhase is None:
            nPhase = self.opts.get_nSeq()
        # Extract trajectory.
        x = self.x
        # Process the key types.
        KeyTypes = [x.defns[k]['Type'] for k in x.cols]
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
        if not os.path.isdir(frun):
            return
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

    # Prepare the mesh for case *i* (if necessary)
    def PrepareMesh(self, i):
        r"""Prepare the mesh for case *i* if necessary
        
        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
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
        fmsh = self.opts.get_MeshCopyFiles(i=i)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Remove the file if necessary
            if os.path.islink(f1):
                os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1):
                continue
            # Link the file.
            if os.path.isfile(f0):
                shutil.copy(f0, f1)
        # Get the names of input files to copy
        fmsh = self.opts.get_MeshLinkFiles(i=i)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Replace 'x.save' -> 'x.restart'
            f1 = f1.replace('save', 'restart')
            # Remove the file if necessary
            if os.path.islink(f1):
                os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1):
                continue
            # Link the file.
            if os.path.isfile(f0):
                os.symlink(f0, f1)
        # -------
        # Cleanup
        # -------
        # Return to original folder
        os.chdir(fpwd)

    # Write configuration file
    def WriteConfig(self, i, fname='Config.xml'):
        r"""Write configuration file
        
        :Call:
            >>> cntl.WriteConfig(i, fname='Config.xml')
        :Inputs:
            *cntl*: :class:`Cntl`
                Overflow control interface
            *i*: :class:`int`
                Case index
            *fname*: {``'Config.xml'``} | :class:`str`
                Name of file to write within run folder
        :Versions:
            * 2016-08-24 ``@ddalle``: Version 1.0
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

    # Write the PBS script.
    def WritePBS(self, i, nPhase=None):
        r"""Write the PBS script(s) for a given case
        
        :Call:
            >>> cntl.WritePBS(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Run index
            *nPhase*: {``None``} | :class:`int`
                Optional maximum phase number
        :Versions:
            * 2014-10-19 ``@ddalle``: Version 1.0
            * 2016-12-14 ``@ddalle``: Version 1.1; add *nPhase* input
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
            nPBS = max(self.opts.get_nSeq(), nPhase)
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
            # Initialize options to `run_overflow.py`
            flgs = ''
            # Get specific python version
            pyexec = self.opts.get_PythonExec(j)
            # Simply call the advanced interface.
            f.write('\n# Call the OVERFLOW interface.\n')
            if pyexec:
                # Use specific version
                f.write("%s -m cape.pyover run %s\n" % (pyexec, flgs))
            else:
                # Use CAPE-provided script
                f.write('run_overflow.py' + flgs + '\n')
            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)
  # >

  # ================
  # Global options
  # ================
  # <
    # Get the project rootname
    def GetPrefix(self, j=0):
        r"""Get the project root name or OVERFLOW file prefix
        
        :Call:
            >>> name = cntl.GetPrefix(j=0)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
        """
        # Return the value from options
        return self.opts.get_Prefix(j)
  # >

  # =============
  # Namelist
  # =============
  # <  
    # Read namelist
    def ReadNamelist(self, j=0, q=True):
        r"""Read the OVERFLOW namelist template
        
        :Call:
            >>> cntl.ReadNamelist(j=0, q=True)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
        """
        # File name
        fnml = self.opts.get_OverNamelist(j)
        # Check for empty value
        if fnml is None:
            return
        # Check for absolute path
        if not os.path.isabs(fnml):
            # Use path relative to JSON root
            fnml = os.path.join(self.RootDir, fnml)
        # Read the file
        nml = OverNamelist(fnml)
        # Save it.
        if q:
            # Read to main slot for modification
            self.Namelist = nml
        else:
            # Template for reading original parameters
            self.Namelist0 = nml
        
    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        r"""Get a namelist variable's value
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> val = cntl.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
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
            * 2016-02-01 ``@ddalle``: Version 1.0
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
  # >

  # ================
  # Mesh
  # ================
  # <
    # Get list of raw file names
    def GetMeshFileNames(self, i=0):
        r"""Return the list of mesh files
        
        :Call:
            >>> fname = cntl.GetMeshFileNames()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names read from root directory
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
        """
        # Get config
        config = self.GetConfig(i)
        # Get the file names from *opts*
        fname = self.opts.get_MeshFiles(i=i)
        # Remove folders
        fname = [os.path.split(f)[-1] for f in fname]
        # Output
        return fname

    # Get the project rootname
    def GetConfig(self, i):
        r"""Get the configuration (if any) for case *i*
        
        If there is no *config* or similar run matrix variable, return the name
        of the group folder
        
        :Call:
            >>> config = cntl.GetConfig(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Case index
        :Outputs:
            *config*: :class:`str`
                Case configuration
        :Versions:
            * 2016-02-02 ``@ddalle``: Version 1.0
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
            config += self.x[k][i]
        # Output
        return config

    # Function to get configuration directory
    def GetConfigDir(self, i):
        r"""Return absolute path to configuration folder
        
        :Call:
            >>> fcfg = cntl.GetConfigDir(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Case index
        :Outputs:
            *fcfg*: :class:`str`
                Full path to configuration folder
        :Versions:
            * 2016-02-02 ``@ddalle``: Version 1.0
            * 2023-03-18 ``@ddalle``: v1.1; :mod:`optdict` changes
        """
        # Configuration of this case
        config = self.GetConfig(i)
        # Configuration folder
        fcfg = self.opts.get_MeshConfigDir()
        # Check if it begins with a slash.
        if os.path.isabs(fcfg):
            # Return as absolute path
            return fcfg
        else:
            # Append the root directory
            return os.path.join(self.RootDir, fcfg)
  # >

  # ====================
  # DataBook
  # ====================
  # <
    # Function to read the databook.
    def ReadDataBook(self, comp=None):
        r"""Read the current data book
        
        :Call:
            >>> cntl.ReadDataBook(comp=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *comp*: {``None``} | :class:`str` | :class:`list`
                List of components, or read all if ``None``
        :Versions:
            * 2016-02-17 ``@ddalle``: Version 1.0
            * 2017-04-27 ``@ddalle``: Version 1l1; add *comp* option
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
        # Ensure list of components
        if comp is not None:
            comp = list(np.array(comp).flatten())
        # Read the data book.
        self.DataBook = dataBook.DataBook(self, comp=comp)
        # Return to original folder.
        os.chdir(fpwd)
  # >

  # ===============
  # Case Interface
  # ===============
  # <
    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        r"""Get the current iteration number from the appropriate module

        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class

        :Call:
            >>> n = cntl.CaseGetCurrentIter()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: Version 1.0
        """
        # Read value
        n = case.GetCurrentIter()
        # Default to zero.
        if n is None:
            return 0
        else:
            return n

    # Check a case's phase output files
    def CheckUsedPhase(self, i, v=False):
        r"""Check maximum phase number run at least once
        
        :Call:
            >>> j, n = cntl.CheckUsedPhase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
            *n*: :class:`int`
                Max phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: Version 1.0
            * 2017-07-11 ``@ddalle``: Version 1.1; add *v*
        """
        # Check input
        if type(i).__name__ not in ["int", "int64", "int32"]:
            raise TypeError(
                "Input to :func:`Cntl.CheckCase()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Initialize phase number.
        j = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            # Verbosity option
            if v: print("    Folder '%s' does not exist" % frun)
            j = None
        # Check that test.
        if j is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Read local settings
            try:
                # Read "case.json"
                rc = case.ReadCaseJSON()
                # Get phase list
                phases = list(self.opts.get_PhaseSequence())
            except Exception:
                # Get global phase list
                phases = list(self.opts.get_PhaseSequence())
            # Reverse the list
            phases.reverse()
            # Loop backwards
            for j in phases:
                # Check if any output files exist
                if len(case.glob.glob("run.%02i.[1-9]*"%(j+1))) > 0:
                    # Found it.
                    break
        # Return to original folder.
        os.chdir(fpwd)
        # Output.
        return j, phases[-1]

    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentPhase(self):
        r"""Get the current phase number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> j = cntl.CaseGetCurrentPhase()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: Version 1.0
        """
        # Be safe
        try:
            # Read the "case.json" folder
            rc = case.ReadCaseJSON()
            # Get the phase number
            return case.GetPhaseNumber(rc)
        except:
            return 0

    # Function to check if the mesh for case *i* is prepared
    def CheckMesh(self, i):
        r"""Check if the mesh for case *i* is prepared
        
        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of OVERFLOW run control class
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
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

    # Check for a failure.
    def CheckError(self, i):
        r"""Check if a case has a failure
        
        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                OVERFLOW control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        :Versions:
            * 2015-01-02 ``@ddalle``: Version 1.0
            * 2017-04-06 ``@ddalle``: Checking for ``q.bomb``
        """
        # Safely go to root.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the FAIL file.
        q = self.x.ERROR[i]
        q = q or os.path.isfile(os.path.join(frun, 'FAIL'))
        q = q or os.path.isfile(os.path.join(frun, 'q.bomb'))
        # Check for 'nan_locations*.dat'
        if not q:
            # Get list of files
            fglob = case.glob.glob(os.path.join(frun, "core.[0-9]*"))
            # Check for any
            q = (len(fglob) > 0)
        # Go home.
        os.chdir(fpwd)
        # Output
        return q

    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v=False):
        r"""Check if the current folder has the necessary files to run
        
        :Call:
            >>> q = cntl.CheckNone(v=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
            *v*: ``True`` | {``False``}
                Verbose flag
        :Outputs:
            *q*: ``True`` | ``False``
        :Versions:
            * 2015-10-19 ``@ddalle``: Version 1.0
            * 2017-02-22 ``@ddalle``: Version 1.1; add *v*
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
            if not (os.path.isfile(fo) or os.path.isfile(fi)):
                # Verbose flag
                if v: print("    Missing file '%s'" % fi)
                return True
        # Apparently no issues.
        return False

    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i, running=False):
        r"""Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = cntl.GetCPUTime(i, running=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                OVERFLOW control interface
            *i*: :class:`int`
                Case index
            *running*: ``True`` | {``False``}
                Whether or not the case is running
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: Version 1.0
            * 2016-08-31 ``@ddalle``: Version 1.1; start times
        """
        # File names
        fname = 'pyover_time.dat'
        fstrt = 'pyover_start.dat'
        # Call the general function using hard-coded file name
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)
        
    # Read run control options from case JSON file
    def ReadCaseJSON(self, i):
        r"""Read ``case.json`` file from case *i* if possible

        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
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
        r"""Read namelist from case *i*, phase *j* if possible
        
        :Call:
            >>> nml = cntl.ReadCaseNamelist(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Run index
            *rc*: ``None`` | :class:`RunControl`
                Run control interface read from ``case.json`` file
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *nml*: ``None`` | :class:`OverNamelist`
                Namelist interface is possible
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
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

    # Stop a set of cases
    def StopCases(self, n=0, **kw):
        r"""Stop one or more cases by writing a ``STOP`` file
        
        :Call:
            >>> cntl.StopCases(n=0, cons=[], I=None, **kw)
        :Inputs:
            *n*: ``None`` | {``0``} | positive :class:`int`
                Iteration at which to stop
            *cons*: :class:`list`\ [:class:`str`]
                List of trajectory constraints
            *I*: :class:`list`\ [:class:`int`]
                List of case indices
        :See also:
            * :func:`cape.runmatrix.RunMatrix.GetIndices`
        :Versions:
            * 2017-03-07 ``@ddalle``: Version 1.0
        """
        # Get list of cases
        I = self.x.GetIndices(**kw)
        # Save current location
        fpwd = os.getcwd()
        # Loop through cases
        for i in I:
            # Run directory
            frun = self.x.GetFullFolderNames(i)
            # Go to rood directory
            os.chdir(self.RootDir)
            # Check if the folder exists
            if not os.path.isdir(frun): continue
            # Otherwise, enter the folder
            os.chdir(frun)
            # Write a STOP file
            case.WriteStopIter(n)
        # Return to original location
        os.chdir(fpwd)
  # >

  # =========================
  # Report; major attributes
  # =========================
  # <
    # Function to read a report
    def ReadReport(self, rep):
        r"""Read a report interface
        
        :Call:
            >>> R = cntl.ReadReport(rep)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of control class containing relevant parameters
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *R*: :class:`cape.pyover.report.Report`
                Report interface
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Read the report
        R = report.Report(self, rep)
        # Output
        return R
  # >

  # =============
  # BCs
  # =============
  # <
    # Prepare surface BC
    def SetSurfBC(self, key, i, CT=False):
        r"""Set a surface BC for one key using *IBTYP* 153
        
        :Call:
            >>> cntl.SetSurfBC(key, i, CT=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*, *T0* directly)
        :Versions:
            * 2016-08-29 ``@ddalle``: Version 1.0
        """
        # Set the trajectory key type to look for
        if CT:
            # Setting BCs by thrust
            typ = "SurfCT"
        else:
            # Setting BCs by pressure/total pressure
            typ = "SurfBC"
        # Get list of grids
        grids = self.x.GetSurfBC_Grids(i, key, typ=typ)
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
            comp = self.x.GetSurfBC_CompID(i, key, comp=grid, typ=typ)
            # BC index
            bci = self.x.GetSurfBC_BCIndex(i, key, comp=grid, typ=typ)
            # Boundary conditions
            if CT:
                # Get *p0*, *T0* from thrust
                p0, T0 = self.GetSurfCTState(key, i, grid)
            else:
                # Use *p0* and *T0* directly as inputs
                p0, T0 = self.GetSurfBCState(key, i, grid)
            # Species
            Y = self.x.GetSurfBC_Species(i, key, comp=grid, typ=typ)
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
                        ("Received column index %s for grid '%s' "%(bci,grid))+
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
        r"""Get stagnation pressure and temperature ratios
        
        :Call:
            >>> p0, T0 = cntl.GetSurfBC(key, i, grid=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
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
            * 2016-08-29 ``@ddalle``: Version 1.0
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
        r"""Get stagnation pressure and temp. ratios for *SurfCT* key

        :Call:
            >>> p0, T0 = cntl.GetSurfCTState(key, i, grid=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
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
            * 2016-08-29 ``@ddalle``: Version 1.0
        """
        # Get the trhust value
        CT = self.x.GetSurfCT_Thrust(i, key, comp=grid)
        # Get the exit parameters
        M2 = self.x.GetSurfCT_ExitMach(i, key, comp=grid)
        A2 = self.x.GetSurfCT_ExitArea(i, key, comp=grid)
        # Ratio of specific heats
        gam = self.x.GetSurfCT_Gamma(i, key, comp=grid)
        # Derivative gas constants
        g2 = 0.5 * (gam-1)
        g3 = gam / (gam-1)
        # Get reference dynamice pressure
        qref = self.x.GetSurfCT_RefDynamicPressure(i, key, comp=grid)
        # Get reference area
        Aref = self.GetSurfCT_RefArea(key, i)
        # Calculate total pressure
        p0 = CT*qref*Aref/A2 *  (1+g2*M2*M2)**g3 / (1+gam*M2*M2)
        # Idenfitifier options
        kwg = {"comp": grid}
        # Temperature inputs
        T0 = self.x.GetSurfCT_TotalTemperature(i, key, **kwg)
        # Calibration
        ap = self.x.GetSurfCT_PressureCalibration(i, key, **kwg)
        bp = self.x.GetSurfCT_PressureOffset(i, key, **kwg)
        aT = self.x.GetSurfCT_TemperatureCalibration(i, key, **kwg)
        bT = self.x.GetSurfCT_TemperatureOffset(i, key, **kwg)
        # Reference values
        p0inf = self.x.GetSurfCT_RefPressure(i, key, **kwg)
        T0inf = self.x.GetSurfCT_RefTemperature(i, key, **kwg)
        # Output
        return (ap*p0+bp)/p0inf, (aT*T0+bT)/T0inf
  # >

  # ==========
  # Archive
  # ==========
  # <
    # Individual case archive function
    def ArchivePWD(self, phantom=False):
        r"""Archive a single case in the current folder
        
        :Call:
            >>> cntl.ArchivePWD(phantom=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2016-12-09 ``@ddalle``: Version 1.0
            * 2017-12-15 ``@ddalle``: Added *phantom* option
        """
        # Archive using the local module
        manage.ArchiveFolder(self.opts, phantom=phantom)

    # Individual case archive function
    def SkeletonPWD(self, phantom=False):
        r"""Delete most files in current folder, leaving only a skeleton
        
        :Call:
            >>> cntl.SkeletonPWD(phantom=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2017-12-14 ``@ddalle``: Version 1.0
        """
        # Archive using the local module
        manage.SkeletonFolder(self.opts, phantom=phantom)

    # Individual case archive function
    def CleanPWD(self, phantom=False):
        r"""Archive a single case in the current folder
        
        :Call:
            >>> cntl.CleanPWD(phantom=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2017-03-10 ``@ddalle``: Version 1.0
            * 2017-12-15 ``@ddalle``: Added *phantom* option
        """
        # Archive using the local module
        manage.CleanFolder(self.opts, phantom=phantom)
  # >

