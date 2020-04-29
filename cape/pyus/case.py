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


# Function to call script or submit.
def StartCase():
    r"""Start a case by either submitting it or calling locally
    
    :Call:
        >>> case.StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: Copied from :mod:`cape.pycart`
        * 2020-04-27 ``@ddalle``: Copied from :mod:`cape.pyus`
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = 0
    #i = GetPhaseNumber(rc)
    # Check qsub status.
    if rc.get_sbatch(i):
        # Get the name of the PBS file
        fpbs = GetPBSScript(i)
        # Submit the Slurm case
        pbs = queue.psbatch(fpbs)
        return pbs
    elif rc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_us3d()


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
            fphase = "run.%02i.%i" % (i, n)
            # Create empty phase file
            with open(fphase, "w") as f:
                pass
            return
    # Prepare for restart if that's appropriate.
    #SetRestartIter(rc)
    # Check if the primal solution has already been run
    if 0 < ntarg or nprev == 0:
        # Get the ``us3d``
        cmdi = cmd.us3d(rc, i=i)
        # Call the command.
        bin.callf(cmdi, f='us3d.out')
        ## Get new iteration number
        #n1 = GetCurrentIter()
        ## Check for lack of progress
        #if n1 <= n:
        #    raise SystemError("Running phase did not advance iteration count.")
    else:
        # No new iterations
        n1 = n


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


# Run ``us3d-prepar``
def RunUS3DGenBC(rc, i):
    r"""Execute ``us3d-genbc``

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
    ierr = bin.us3d_genbc(rc, i)


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


# Function to determine which PBS script to call
def GetPBSScript(i=None):
    r"""Determine the file name of the PBS script to call
    
    This is a compatibility function for cases that do or do not have
    multiple PBS scripts in a single run directory
    
    :Call:
        >>> fpbs = case.GetPBSScript(i=None)
    :Inputs:
        *i*: :class:`int`
            Run index
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: First version
        * 2020-04-27 ``@ddalle``: US3D version
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if i is not None:
        # Create the name.
        fpbs = 'run_us3d.%02i.pbs' % i
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return 'run_us3d.pbs'
    else:
        # Do not search for numbered PBS script if *i* is None
        return 'run_us3d.pbs'

