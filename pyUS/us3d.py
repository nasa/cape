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
    *cntl.x*               :class:`pyUS.trajectory.Trajectory`
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
from .trajectory import Trajectory

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
        *cntl.x*: :class:`pyFun.trajectory.Trajectory`
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
        self.x = Trajectory(**self.opts['Trajectory'])

        # Job list
        self.jobs = {}
        
        # Read the input file template(s)
        #self.ReadInputInp()
        
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
    
