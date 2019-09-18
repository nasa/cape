"""
:mod:`pyUS.case`: Case control module
======================================

This module contains the important function :func:`case.run_us3d`, which
actually runs ``us3d`` or ``us3d_mpi``, along with the utilities that
support it.

It also contains US3D-specific versions of some of the generic methods from
:mod:`cape.case`.  For instance the function :func:`GetCurrentIter` determines
how many US3D iterations have been run in the current folder, which is
obviously a solver-specific task.


"""

# Standard library modules
import os
import glob
import json
import shutil
import resource

# Direct imports from standard modules
from datetime import datetime

# CAPE modules
import cape.case
import cape.manage

# Local imports
#from . import bin
#from . import cmd
#from . import queue

# Partial local imports
from .options.runControl import RunControl
from .inputInp import InputInp



# Function to complete the setup and call appropriate US3D commands
def run_us3d():
    """Setup and run the appropriate US3D command
    
    :Call:
        >>> pyUS.case.run_us3d()
    :Versions:
        * 2019-06-27 ``@ddalle``: Started
    """
    # Check for RUNNING file.
    if os.path.isfile('RUNNING'):
        # Case already running
        raise IOError('Case already running!')
    # Touch (create) the running file
    open("RUNNING", "w").close()
    # Start timer
    tic = datetime.now()
    # Get the run control settings
    rc = ReadCaseJSON()
    # Determine the run index.
    #i = GetPhaseNumber(rc)
    # Write the start time
    WriteStartTime(tic, rc, i)
    # Prepare files
    #PrepareFiles(rc, i)
    # Prepare environment variables (other than OMP_NUM_THREADS)
    cape.case.PrepareEnvironment(rc, i)
    # Run the appropriate commands
    #RunPhase(rc, i)
    # Clean up files
    #FinalizeFiles(rc, i)
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'):
        os.remove('RUNNING')
    # Save time usage
    WriteUserTime(tic, rc, i)
    # Check for errors
    #CheckSuccess(rc, i)
    # Resubmit/restart if this point is reached.
    #RestartCase(i)


# Function to read the local settings file.
def ReadCaseJSON():
    """Read `RunControl` settings for local case
    
    :Call:
        >>> rc = pyUS.case.ReadCaseJSON()
    :Outputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control settings
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
    """
    # Read the file, fail if not present.
    with open('case.json') as f:
        # Read the settings
        opts = json.load(f)
    # Convert to a flowCart object.
    rc = RunControl(**opts)
    # Output
    return rc


# Interpret the input.inp file for a phase
def GetInputInp(rc=None, j=None):
    """Read case namelist ``input.inp`` file
    
    :Call:
        >>> inp = pyUS.case.GetInputInp(rc=None, j=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *j*: :class:`int`
            Phase number
    :Outputs:
        *inp*: :class:`pyUS.inputInp.InputInp`
            Interface to ``input.inp``
    :Versions:
        * 2019-06-27 ``@ddalle``: First version
    """
    # Process phase number
    if j is None and rc is not None:
        # Default to most recent phase number
        j = GetPhaseNumber(rc)
    # Check for folder with no working ``case.json``
    if rc is None:
        # Check for simplest namelist file
        if os.path.isfile('input.inp'):
            # Read the currently linked input file
            inp = InputInp('input.inp')
        else:
            # Look for input.inp files
            fglob = glob.glob('input.[0-9][0-9].inp')
            # Sort it
            fglob.sort()
            # Read last input file when sorted
            inp = InputInp(fglob[-1])
        # Output
        return inp
    # Get the directly specified file
    inp = InputInp('input.%02i.inp' % j)
    # Output
    return inp


# Write start time
def WriteStartTime(tic, rc, i, fname="pyus_start.dat"):
    """Write the start time in *tic*
    
    :Call:
        >>> WriteStartTime(tic, rc, i, fname="pyus_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: {``"pyus_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2016-08-31 ``@ddalle``: First version
        * 2019-06-27 ``@ddalle``: US3D version
    """
    # Call the function from :mod:`cape.case`
    cape.case.WriteStartTimeProg(tic, rc, i, fname, 'run_us3d.py')

    
# Write time used
def WriteUserTime(tic, rc, i, fname="pyus_time.dat"):
    """Write time usage since time *tic* to file
    
    :Call:
        >>> toc = WriteUserTime(tic, rc, i, fname="pyus_time.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: :class:`str`
            Name of file containing CPU usage history
    :Outputs:
        *toc*: :class:`datetime.datetime`
            Time at which time delta was measured
    :Versions:
        * 2015-12-09 ``@ddalle``: First version
        * 2019-06-27 ``@ddalle``: US3D version
    """
    # Call the function from :mod:`cape.case`
    cape.case.WriteUserTimeProg(tic, rc, i, fname, 'run_us3d.py')

