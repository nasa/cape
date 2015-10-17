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

# pyCart settings class
from . import options

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
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyFun.Fun3d(nCase=%i)>" % (
            self.x.nCase)
        
        
        
    # Read the namelist
    def ReadNamelist(self):
        """Read the :file:`fun3d.nml` file
        
        :Call:
            >>> fun3d.ReadInputCntl()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of the pyFun control class
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        # CHange to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file.
        self.Namelist = Namelist(self.opts.get_Namelist())
        # Go back to original location
        os.chdir(fpwd)
        
        
# class Fun3d

