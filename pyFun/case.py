"""
Case control module: :mod:`pyFun.case`
======================================

This module contains functions to execute FUN3D and interact with individual
case folders.
"""

# Import options class
from options.runControl import RunControl
# Import the namelist
from namelist import Namelist
# Interface for writing commands
from . import bin


# Read the local JSON file
import json
# File control
import os, glob, shutil
# Basic numerics
from numpy import nan, isnan



# Function to complete final setup and call the appropriate FUN3D commands
def run_fun3d():
    pass


# Function to determine which PBS script to call
def GetPBSScript(i=None):
    """Determine the file name of the PBS script to call
    
    This is a compatibility function for cases that do or do not have multiple
    PBS scripts in a single run directory
    
    :Call:
        >>> fpbs = pyFun.case.GetPBSScript(i=None)
    :Inputs:
        *i*: :class:`int`
            Run index
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if i is not None:
        # Create the name.
        fpbs = 'run_fun3d.%02i.pbs' % i
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return 'run_fun3d.pbs'
    else:
        # Do not search for numbered PBS script if *i* is None
        return 'run_fun3d.pbs'
    

# Get the namelist
def GetNamelist(rc=None):
    """Read case namelist file
    
    :Call:
        >>> nml = pyFun.case.GetNamelist(rc=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
    :Outputs:
        *nml*: :class:`pyFun.namelist.Namelist`
            Namelist interface
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    if rc is None:
        # Check for simplest namelist file
        if os.path.isfile('fun3d.nml'):
            # Read the currently linked namelist.
            return Namelist('fun3d.nml')
        else:
            # Look for namelist files
            fglob = glob.glob('fun3d.??.nml')
            # Read one of them.
            return Namelist(fglob[0])
    else:
        pass


# Get the project rootname
def GetProjectRootname():
    """Read namelist and return project namelist
    
    :Call:
        >>> rname = pyFun.case.GetProjectRootname()
    :Outputs:
        *rname*: :class:`str`
            Project rootname
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read a namelist.
    nml = GetNamelist()
    # Read the project root name
    return nml.GetRootname()
    
    

# Function to read the local settings file.
def ReadCaseJSON():
    """Read `RunControl` settings for local case
    
    :Call:
        >>> rc = pyFun.case.ReadCaseJSON()
    :Outputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control settings
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
    """
    # Read the file, fail if not present.
    f = open('case.json')
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a flowCart object.
    rc = RunControl(**opts)
    # Output
    return rc
    

# Get last line of 'history.dat'
def GetCurrentIter():
    """Get the most recent iteration number
    
    :Call:
        >>> n = pyFun.case.GetHistoryIter()
    :Outputs:
        *n*: :class:`float`
            Last iteration number
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read the project rootname
    try:
        rname = GetCurrentIter()
    except Exception:
        # No iterations
        return nan
    # Assemble file name.
    fname = "%s_hist.dat" % rname
    # Check for the file.
    if not os.path.isfile(fname):
        # No history to read.
        return nan
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
        # Get the iteration number.
        return float(txt.split()[0])
    except Exception:
        # Failure; return no-iteration result.
        return nan


