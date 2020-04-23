"""
:mod:`cape.pyus.case`: Case control module
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
import cape.cfdx.case as cc
import cape.cfdx.queue as queue
import cape.manage

# Local imports
from . import bin
from . import cmd

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
    i = GetPhaseNumber(rc)
    # Write the start time
    WriteStartTime(tic, rc, i)
    # Prepare files
    PrepareFiles(rc, i)
    # Prepare environment variables (other than OMP_NUM_THREADS)
    cc.PrepareEnvironment(rc, i)
    # Run the appropriate commands
    RunPhase(rc, i)
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
        *rc*: :class:`cape.pyus.options.runControl.RunControl`
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
        *rc*: :class:`cape.pyus.options.runControl.RunControl`
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


# Prepare the files of the case
def PrepareFiles(rc, i=None):
    r"""Prepare file names appropriate to run phase *i* of FUN3D
    
    :Call:
        >>> PrepareFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`cape.pyus.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-04-14 ``@ddalle``: First version
    """
    # Get the phase number if necessary
    if i is None:
        # Get the phase number
        i = GetPhaseNumber(rc)
    # Delete any input file (primary namelist)
    if os.path.isfile('input.inp') or os.path.islink('input.inp'):
        os.remove('input.inp')
    # Create the correct namelist
    os.symlink('input.%02i.inp' % i, 'input.inp')


# Function to chose the correct input to use from the sequence.
def GetPhaseNumber(rc):
    r"""Determine the phase number based on files in folder
    
    :Call:
        >>> i = case.GetPhaseNumber(rc)
    :Inputs:
        *rc*: :class:`cape.pyfun.options.runControl.RunControl`
            Options interface for run control
    :Outputs:
        *i*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2020-10-02 ``@ddalle``: Placeholder version
    """
    # Just assume first phase
    return 0


# Run one phase appropriately
def RunPhase(rc, i):
    r"""Run one phase using appropriate commands
    
    :Call:
        >>> RunPhase(rc, i)
    :Inputs:
        *rc*: :class:`cape.pyus.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2020-04-16 ``@ddalle``: Starter version
    """
    # Count number of times this phase has been run previously.
    nprev = len(glob.glob('run.%02i.*' % i))
    # Read input file
    inp = GetInputInp(rc, i)
    # Get the last iteration number
    #n = GetCurrentIter()
    n = None
    # Number of requested iters for the end of this phase
    ntarg = rc.get_PhaseIters(i)
    # Number of iterations to run this phase
    ni = rc.get_nIter(i)
    # Mesh generation and verification actions
    if i == 0 and n is None:
        # Prepare files
        RunUS3DPrepar(rc, i)
        # Check for mesh-only phase
        if not (ntarg and ni):
            # Make sure *n* is not ``None``
            if n is None:
                n = 0
            # Create an output file to make phase number programs work
            os.system('touch run.%02i.%i' % (i, n))
            return
    # Prepare for restart if that's appropriate.
    #SetRestartIter(rc)
    return
    # Check if the primal solution has already been run
    if n < ntarg or nprev == 0:
        # Get the ``us3d``
        cmdi = cmd.us3d(rc, i=i)
        # Call the command.
        bin.callf(cmdi, f='fun3d.out')
        # Get new iteration number
        n1 = GetCurrentIter()
        # Check for lack of progress
        if n1 <= n:
            raise SystemError("Running phase did not advance iteration count.")
    else:
        # No new iteratoins
        n1 = n
    # Go back up a folder if we're in the "Flow" folder
    if rc.get_Dual(): os.chdir('..')
    # Check current iteration count.
    if (i >= rc.get_PhaseSequence(-1)) and (n >= rc.get_LastIter()):
        return
    # Check for adaptive solves
    if n1 < np: return
    # Check for adjoint solver
    if rc.get_Dual() and rc.get_DualPhase(i):
        # Copy the correct namelist
        os.chdir('Flow')
        # Delete ``fun3d.nml`` if appropriate
        if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
            os.remove('fun3d.nml')
        # Copy the correct one into place
        os.symlink('fun3d.dual.%02i.nml' % i, 'fun3d.nml')
        # Enter the 'Adjoint/' folder
        os.chdir('..')
        os.chdir('Adjoint')
        # Create the command to calculate the adjoint
        cmdi = cmd.dual(rc, i=i, rad=False, adapt=False)
        # Run the adjoint analysis
        bin.callf(cmdi, f='dual.out')
        # Create the command to adapt
        cmdi = cmd.dual(rc, i=i, adapt=True)
        # Estimate error and adapt
        bin.callf(cmdi, f='dual.out')
        # Rename output file after completing that command
        os.rename('dual.out', 'dual.%02i.out' % i)
        # Return
        os.chdir('..')
    elif rc.get_Adaptive() and rc.get_AdaptPhase(i):
        # Check if this is a weird mixed case with Dual and Adaptive
        if rc.get_Dual(): os.chdir('Flow')
        # Run the feature-based adaptive mesher
        cmdi = cmd.nodet(rc, adapt=True, i=i)
        # Make sure "restart_read" is set to .true.
        nml.SetRestart(True)
        nml.Write('fun3d.%02i.nml' % i)
        # Call the command.
        bin.callf(cmdi, f='adapt.out')
        # Rename output file after completing that command
        os.rename('adapt.out', 'adapt.%02i.out' % i)
        # Return home if appropriate
        if rc.get_Dual(): os.chdir('..')


# Run ``us3d-prepar``
def RunUS3DPrepar(rc, i):
    r"""Execute ``us3d-prepar``

    :Call:
        >>> RunUS3DPrepar(rc, i)
    :Inputs:
        *rc*: :class:`cape.pyus.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number, does nothing if *i* is not ``0``
    :Versions:
        * 2020-04-16 ``@ddalle``: First version
    """
    # Get phase number
    if i != 0:
        # Do nothing
        return
    # Execute command
    return bin.us3d_prepar(rc, i)


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
    cc.WriteStartTimeProg(tic, rc, i, fname, 'run_us3d.py')

    
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
    cc.WriteUserTimeProg(tic, rc, i, fname, 'run_us3d.py')

