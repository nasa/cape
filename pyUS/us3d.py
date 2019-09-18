#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`pyUS.us3d`: US3D control module 
=========================================

This module provides tools to quickly setup basic or complex US3D run matrices
and serve as an executive for pre-processing, running, post-processing, and
managing the solutions. A collection of cases combined into a run matrix can be
loaded using the following commands.

    .. code-block:: pycon
    
        >>> import pyUS.us3d
        >>> cntl = pyUS.us3d.US3D("pyUS.json")
        >>> cntl
        <pyUS.US3D(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'
        
        
An instance of this :class:`pyUS.us3d.US3D` class has many methods, which
include the run matrix (``cntl.x``), the options interface (``cntl.opts``),
and optionally the data book (``cntl.DataBook``), the appropriate input files
(such as ``cntl.InputInp``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`pyUS.runmatrix.RunMatrix`
    *cntl.opts*            :class:`pyUS.options.Options`
    *cntl.DataBook*        :class:`pyUS.dataBook.DataBook`
    *cntl.InputInp*        :class:`pyUS.inputInp.InputInp`
    ====================   =============================================

Finally, the :class:`pyUs.us3d.US3D` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE class are
also available here.

"""

# System modules
import os
import json
import shutil
import subprocess as sp

# Standard library direct inports
from datetime import datetime

# Third-party modules
import numpy as np

# Unmodified CAPE modules
from cape import convert

# CAPE classes and specific imports
from cape.cntl import Cntl
from cape.util import RangeString

# Full pyUS modules
from . import options
#from . import manage
from . import case
#from . import mapbc
#from . import faux
#from . import dataBook
#from . import report

# Functions and classes from local modules
from .inputInp   import InputInp
from .runmatrix import RunMatrix

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyUSFolder = os.path.split(_fname)[0]
    
# Class to read input files
class US3D(Cntl):
    """
    Class for handling global options and setup for US3D.
    
    This class is intended to handle all settings used to describe a group
    of US3D cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha sweep
    for each single control variable setting.
    
    The settings are read from a JSON file, which is robust and simple to
    read.  The settings can be extended using Python modules if desired.
    
    Defaults are read from the file ``$CAPE/settings/pyUS.default.json``.
    
    :Call:
        >>> cntl = pyFun.Fun3d(fname="pyFun.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyUS input file
    :Outputs:
        *cntl*: :class:`pyFun.fun3d.Fun3d`
            Instance of the pyUS control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`pyFun.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2019-06-04 ``@ddalle``: Started
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname="pyUS.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No pyUS control file '%s' found" % fname)
            
        # Get the real path
        fjson = os.path.realpath(fname)
        # Save it
        self.fname = fjson
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.ImportModules()
        
        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}
        
        # Read the input file template(s)
        self.ReadInputInp()
        
        # Read the configuration
        #self.ReadConfig()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
        # Run any initialization functions
        self.InitFunction()
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return '<pyUS.US3D("%s", nCase=%i)>' % (
            os.path.split(self.fname)[1],
            self.x.nCase)
  # >
  
  
  # ================
  # File: input.inp
  # ================
  # <
    # Read the namelist
    def ReadInputInp(self, j=0, q=True):
        """Read the ``input.inp`` template file
        
        :Call:
            >>> cntl.InputInp(j=0, q=True)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *j*: :class:`int`
                Phase number
            *q*: {``True``} | ``False``
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2015-12-31 ``@ddalle``: Added *Namelist0*
            * 2019-06-27 ``@ddalle``: Forked from :func:`ReadNamelist`
        """
        # template file name
        finp = self.opts.get_InputInp(j)
        # Check for empty value
        if finp is None:
            return
        # Check for absolute path
        if not os.path.isabs(finp):
            # Use path relative to JSON root
            finp = os.path.join(self.RootDir, finp)
        # Read the file
        inp = InputInp(finp)
        # Save it.
        if q:
            # Read to main slot for modification
            self.InputInp = inp
        else:
            # Template for reading original parameters
            self.NamelisInputInp0 = inp
    
    # Get value from ``input.inp`` key
    def GetInputInpVar(self, sec, key, j=0):
        """Get a ``input.inp`` variable's value
        
        The JSON file overrides the value from the template file
        
        :Call:
            >>> val = cntl.getInputVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*: :class:`any`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From :func:`GetNamelistVar`
        """
        # Get the value from template
        tval = self.InputInp.GetVar(sec, key)
        # Check for options value.
        if tval is None:
            # No template file value
            return self.opts.get_InputInp_key(sec, key, j)
        elif 'US3D' not in self.opts:
            # No section in options
            return tval
        elif sec not in self.opts['US3D']:
            # No corresponding options section
            return tval
        elif key not in self.opts['US3D'][sec]:
            # Value not specified in the options namelist
            return tval
        else:
            # Default to the options
            return self.opts.get_InputInp_key(sec, key, j)
  # >
  
  
  # ============
  # Case Status
  # ============
  # <
  
  
            
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i, running=False):
        """Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = cntl.GetCPUTime(i, running=False)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Case index
            *runing*: ``True`` | {``False``}
                Whether or not to check for time since last start
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
            * 2016-08-31 ``@ddalle``: Checking time since last start
        """
        # File names
        fname = 'pyus_time.dat'
        fstrt = 'pyus_start.dat'
        # Call the general function using hard-coded file name
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)
        
  # >
  
  
  
  
  # ==============
  # Case Interface
  # ==============
  # <
    # Read run control options from case JSON file
    def ReadCaseJSON(self, i):
        """Read ``case.json`` file from case *i* if possible
        
        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`pyOver.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
            * 2017-03-31 ``@ddalle``: Copied from :mod:`pyOver`
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
        
    # Read ``input.inp`` file from a case folder
    def ReadCaseInputInp(self, i, rc=None, j=None):
        """Read ``input.inp`` from case *i*, phase *j* if possible
        
        :Call:
            >>> inp = cntl.ReadCaseInputInp(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Case index
            *rc*: ``None`` | :class:`pyUS.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *inp*: ``None`` | :class:`pyUS.inputInp.InputInp`
                US3D input interface if possible
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From pyFun
        """
        # Read the *rc* if necessary
        if rc is None:
            rc = self.ReadCaseJSON(i)
        # If still None, exit
        if rc is None:
            return
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
        nml = case.GetInputInp(rc, j)
        # Return to original location
        os.chdir(fpwd)
        # Output
        return nml
        
    # Write the PBS script.
    def WritePBS(self, i):
        """Write the PBS script(s) for a given case
        
        :Call:
            >>> cntl.WritePBS(i)
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-10-19 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: US3D version
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
                fpbs = 'run_us3d.%02i.pbs' % j
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_us3d.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)
            
            # Initialize options to `run_FUN3D.py`
            flgs = ''

            # Simply call the advanced interface.
            f.write('\n# Call the US3D interface.\n')
            f.write('run_us3d.py' + flgs + '\n')
            
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
            >>> pbs = cntl.CaseStartCase()
        :Inputs:
            *cntl*: :class:`pyUS.us3d.US3D`
                US3D control interface
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: US3D version
        """
        return case.StartCase()
  # >
  
# class US3D

