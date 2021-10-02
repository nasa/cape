"""
:mod:`cape.pyover.case`: OVERFLOWase control module
=====================================================

This module contains the important function :func:`case.run_overflow`, which
actually runs ``overrunmpi`` or whichever executable is specified by the user,
along with the utilities that support it.

It also contains OVERFLOW-specific versions of some of the generic methods from
:mod:`cape.case`. For instance the function :func:`GetCurrentIter` determines
how many OVERFLOW iterations have been run in the current folder, which is
obviously a solver-specific task. It also contains the function :func:`LinkQ`
and :func:`LinkX` which creates links to fixed file names from the most recent
output created by OVERFLOW, which is useful for creating simpler Tecplot
layouts, for example.

All of the functions from :mod:`cape.case` are imported here.  Thus they are
available unless specifically overwritten by specific :mod:`cape.pyover` versions.

"""

# Standard library modules
import os
import glob
import json
import shutil
import resource
import re

# Standard library direct imports
from datetime import datetime

# Numerics
import numpy as np

# Template CAPE module
import cape.cfdx.case as cc

# Import options class
from .options.runControl import RunControl
# Import the namelist
from .overNamelist import OverNamelist
# Interface for writing commands
from . import bin
from . import cmd
from .. import queue

global twall, dtwall, twall_avail

# Total wall time used
twall = 0.0
# Time used by last phase
dtwall = 0.0
# Default time avail
twall_avail = 1e99



# Function to complete final setup and call the appropriate FUN3D commands
def run_overflow():
    """Setup and run the appropriate OVERFLOW command
    
    :Call:
        >>> pyFun.case.run_overflow()
    :Versions:
        * 2016-02-02 ``@ddalle``: First version
    """
    # Check for RUNNING file.
    if os.path.isfile('RUNNING'):
        # Case already running
        raise IOError('Case already running!')
    # Start timer
    tic = datetime.now()
    # Get the run control settings
    rc = ReadCaseJSON()
    # Get the project name
    fproj = GetPrefix()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Write start time
    WriteStartTime(tic, rc, i)
    # Delete any input file.
    if os.path.isfile('over.namelist') or os.path.islink('over.namelist'):
        os.remove('over.namelist')
    # Prepare environment variables (other than OMP_NUM_THREADS)
    cc.PrepareEnvironment(rc, i)
    # Create the correct namelist.
    shutil.copy('%s.%02i.inp' % (fproj,i+1), 'over.namelist')
    # Get the `nodet` or `nodet_mpi` command
    cmdi = cmd.overrun(rc, i=i)
    # Call the command.
    bin.callf(cmdi, f='overrun.out', check=False)
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Save time usage
    WriteUserTime(tic, rc, i)
    # Get the most recent iteration number
    n = GetCurrentIter()
    # Get STOP iteration, if any
    nstop = GetStopIter()
    # Assuming that worked, move the temp output file.
    fout = '%s.%02i.out' % (fproj, i+1)
    if os.path.isfile(fout):
        os.rename(fout, '%s.%02i.%i' % (fproj, i+1, n))
    # Check current iteration count and phase
    if (i>=rc.get_PhaseSequence(-1)) and (n>=rc.get_LastIter()):
        # Case completed
        return
    elif (nstop is not None) and (n >= nstop):
        # Stop requested externally
        return
    # Resubmit/restart if this point is reached.
    RestartCase(i)
# def run_overflow

# Function to call script or submit.
def StartCase():
    """Start a case by either submitting it or calling with a system command
    
    :Call:
        >>> pyOver.case.StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: Copied from pyCart
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Check qsub status.
    if rc.get_sbatch(i):
        # Getthe name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case
        pbs = queue.sbatch(fpbs)
        return pbs
    elif rc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_overflow()
        
# Get STOP iteration
def GetStopIter():
    """Get iteration at which to stop by reading ``STOP`` file
    
    If the file exists but is empty, returns ``0``; if file does not exist,
    returns ``None``; and otherwise reads the iteration number from the file.
    
    :Call:
        >>> n = pyOver.case.GetStopIter()
    :Outputs:
        *n*: ``None`` |  :class:`int`
            Iteration at which to stop OVERFLOW
    :Versions:
        * 2017-03-07 ``@ddalle``: First version
    """
    # Check for the file
    if not os.path.isfile("STOP"):
        # No STOP requested
        return
    # Otherwise, attempt to read it
    try:
        # Open the file
        f = open("STOP", "r")
        # Read the first line
        line = f.readline()
        # Attempt to get an integer out of there
        n = int(line.split()[0])
        return n
    except Exception:
        # If empty file (or not readable), always stop
        return 0
        
# Function to write STOP file
def WriteStopIter(n=0):
    """Create a ``STOP`` file and optionally set the iteration at which to stop
    
    :Call:
        >>> pyOver.case.WriteStopIter(n)
    :Inputs:
        *n*: ``None`` | {``0``} | positive :class:`int`
            Iteration at which to stop; empty file if ``0`` or ``None``
    :Versions:
        * 2017-03-07 ``@ddalle``: First version
    """
    # Create the STOP file
    f = open("STOP", "w")
    # Check if writing anything
    if (n is not None) and (n > 1):
        f.write("%i\n" % n)
    # Close the file
    f.close()
        
# Function to call script or submit
def RestartCase(i0=None):
    """Restart a case by either submitting it or calling with a system command
    
    This version of the command is called with :func:`run_overflow` after
    running a phase or attempting to run a phase.
    
    :Call:
        >>> pyOver.case.RestartCase(i0=None)
    :Inputs:
        *i0*: :class:`int` | ``None``
            Phase index of the previous run
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    global twall, dtwall, twall_avail
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Task manager
    qpbs = rc.get_qsub(i)
    qslr = rc.get_sbatch(i)
    # Get restartability option
    qtime = (twall_avail > twall + dtwall)
    # Status updates: available time
    if twall_avail < 1e6:
        print("   Available time: %.2f hrs" % (twall_avail/3600.0))
    # Used time
    print("   Wall time used: %.2f hrs" % (twall/3600.0))
    print("   Previous phase: %.2f hrs" % (dtwall/3600.0))
    # Don't check time if moving to new phase
    qtime = qtime or (i0 is not None and i0!=i)
    # Check qsub status.
    if not (qpbs or qslr):
        # Run the case.
        run_overflow()
    elif rc.get_Resubmit(i):
        # Check for continuance
        if (i0 is None) or (i>i0) or (not rc.get_Continue(i)):
            # Get the name of the PBS file.
            fpbs = GetPBSScript(i)
            # Submit the case.
            if qslr:
                # Slurm
                pbs = queue.psbatch(fpbs)
            elif qpbs:
                # PBS
                pbs = queue.pqsub(fpbs)
            else:
                # No task manager
                raise NotImplementedError("Could not determine task manager")
            return pbs
        else:
            # Continue on the same job
            if qtime: run_overflow()
    else:
        # Simply run the case. Don't reset modules either.
        if qtime: run_overflow()
    
# Extend case
def ExtendCase(m=1, run=True):
    """Extend the maximum number of iterations and restart/resubmit
    
    :Call:
        >>> ExtendCase(m=1, run=True)
    :Inputs:
        *m*: {``1``} | :class:`int`
            Number of additional sets to add
        *run*: {``True``} | ``False``
            Whether or not to actually run the case
    :Versions:
        * 2016-09-19 ``@ddalle``: First version
    """
    # Check for "RUNNING"
    if os.path.isfile('RUNNING'): return
    # Get the current inputs
    rc = ReadCaseJSON()
    # Get current number of iterations
    n = GetCurrentIter()
    # Check if we have run at least as many iterations as requested
    if n < rc.get_LastIter(): return
    # Get phase number
    j = GetPhaseNumber(rc)
    # Read the namelist
    nml = GetNamelist(rc)
    # Get the number of steps in a phase subrun
    NSTEPS = nml.GetKeyFromGroupName('GLOBAL', 'NSTEPS')
    # Add this count (*m* times) to the last iteration setting
    rc.set_PhaseIters(n+m*NSTEPS, j)
    # Rewrite the settings
    WriteCaseJSON(rc)
    # Start the case if appropriate
    if run:
        StartCase()
    
    
# Write time used
def WriteUserTime(tic, rc, i, fname="pyover_time.dat"):
    """Write time usage since time *tic* to file
    
    :Call:
        >>> toc = WriteUserTime(tic, rc, i, fname="pyover.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: :class:`str`
            Name of file containing CPU usage history
    :Outputs:
        *toc*: :class:`datetime.datetime`
            Time at which time delta was measured
    :Versions:
        * 2015-12-29 ``@ddalle``: First version
    """
    global twall, dtwall
    # Call the function from :mod:`cape.case`
    cc.WriteUserTimeProg(tic, rc, i, fname, 'run_overflow.py')
    # Modify the total time used
    try:
        # Get the result
        A = np.loadtxt(fname, comments='#', usecols=(0,1), delimiter=',')
        # Split out last two entries
        t, n = A.flatten()[-2:]
        # Add to wall time used
        dtwall = 3600.0*t/n
        twall += dtwall
        print("   Wall time used: %.2f hrs (phase %i)" % (dtwall/3600.0, i))
    except Exception:
        # Unknown time
        dtwall = 0.0
        print("   Wall time used: ??? hrs (phase %i)" % i)
        pass

# Read wall time
def ReadWallTimeUsed(fname='pyover_time.dat'):
    global twall, dtwall
    try:
        A = np.loadtxt(fname, comments='#', usecols=(0,1), delimiter=",")
        t,n = A.flatten()[-2:]

        dtwall = 3600.0*t/n
        twall += dtwall
        return dtwall
    except Exception:
        return 0.0

# Write start time
def WriteStartTime(tic, rc, i, fname="pyover_start.dat"):
    """Write the start time in *tic*
    
    :Call:
        >>> WriteStartTime(tic, rc, i, fname="pyover_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: {``"pyover_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2016-08-31 ``@ddalle``: First version
    """
    # Call the function from :mod:`cape.case`
    cc.WriteStartTimeProg(tic, rc, i, fname, 'run_overflow.py')
    
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
        fpbs = 'run_overflow.%02i.pbs' % (i+1)
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return 'run_overflow.pbs'
    else:
        # Do not search for numbered PBS script if *i* is None
        return 'run_overflow.pbs'
    
# Function to chose the correct input to use from the sequence.
def GetPhaseNumber(rc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyOver.case.GetPhaseNumber(rc)
    :Inputs:
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface for run control
    :Outputs:
        *i*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-12-29 ``@ddalle``: FUN3D version
        * 2016-02-03 ``@ddalle``: OVERFLOW version
        * 2017-01-13 ``@ddalle``: Removed req't to have full ``run.%02.*`` seq
    """
    # Get the run index.
    n = GetRestartIter(rc)
    # Initialize list of phases with adequate iters
    JIter = []
    # Initialize list of phases with detected STDOUT files
    JRun = []
    # Loop through possible input numbers.
    for j in range(rc.get_nSeq()):
        # Get the actual run number
        i = rc.get_PhaseSequence(j)
        # Output file glob
        fglob = '%s.%02i.[0-9]*' % (rc.get_Prefix(i), i+1)
        # Check for output files.
        if len(glob.glob(fglob)) > 0:
            # This run has an output file
            JRun.append(j)
        # Check the iteration number.
        if n >= rc.get_PhaseIters(j):
            # The iterations are adequate for this phase
            JIter.append(j)
    # Get phase numbers from the two types
    if len(JIter) > 0:
        jIter = max(JIter) + 1
    else:
        jIter = 0
    if len(JRun) > 0:
        jRun = max(JRun) + 1
    else:
        jRun = 0
    # Look for latest phase with both criteria
    j = min(jIter, jRun)
    # Convert to phase number
    return rc.get_PhaseSequence(j)

# Get the namelist
def GetNamelist(rc=None, i=None):
    """Read case namelist file
    
    :Call:
        >>> nml = pyOver.case.GetNamelist(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *i*: {``None``} | nonnegative :class:`int`
            Phase number (0-based)
    :Outputs:
        *nml*: :class:`pyOver.overNamelist.OverNamelist`
            Namelist interface
    :Versions:
        * 2015-12-29 ``@ddalle``: First version
        * 2015-02-02 ``@ddalle``: Copied from :mod:`cape.pyfun.case`
        * 2016-12-12 ``@ddalle``: Added phase as optional input
    """
    # Check for detailed inputs
    if rc is None:
        # Check for simplest namelist file
        if os.path.isfile('over.namelist'):
            # Read the currently linked namelist.
            return OverNamelist('over.namelist')
        else:
            # Look for namelist files
            fglob = glob.glob('*.[0-9][0-9].inp')
            # Read one of them.
            return OverNamelist(fglob[0])
    else:
        # Get phase number
        if i is None:
            i = GetPhaseNumber(rc)
        # Read the namelist file.
        return OverNamelist('%s.%02i.inp' % (rc.get_Prefix(i), i+1))


# Function to get prefix
def GetPrefix(rc=None, i=None):
    """Read OVERFLOW file prefix
    
    :Call:
        >>> rname = pyFun.case.GetPrefix()
        >>> rname = pyFun.case.GetPrefix(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *i*: :class:`int`
            Phase number
    :Outputs:
        *rname*: :class:`str`
            Project prefix
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    # Get the options if necessary
    if rc is None:
        rc = ReadCaseJSON()
    # Read the prefix
    return rc.get_Prefix(i)
    
# Function to read the local settings file.
def ReadCaseJSON():
    """Read `RunControl` settings for local case
    
    :Call:
        >>> rc = pyOver.case.ReadCaseJSON()
    :Outputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control settings
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-12-29 ``@ddalle``: OVERFLOW version
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
    
# (Re)write the local settings file.
def WriteCaseJSON(rc):
    """Write or rewrite ``RunControl`` settings to ``case.json``
    
    :Call:
        >>> pyOver.case.WriteCaseJSON(rc)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control settings
    :Versions:
        * 2016-09-19 ``@ddalle``: First version
    """
    # Open the file for rewrite
    f = open('case.json', 'w')
    # Dump the Overflow and other run settings.
    json.dump(rc, f, indent=1)
    # Close the file
    f.close()
    

# Get last line of 'history.dat'
def GetCurrentIter():
    """Get the most recent iteration number
    
    :Call:
        >>> n = pyOver.case.GetHistoryIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Last iteration number
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read the two sources
    nh = GetHistoryIter()
    nr = GetRunningIter()
    no = GetOutIter()
    # Process
    if nr is None and no is None:
        # No running iterations; check history
        return nh
    elif nr is None:
        # Intermediate step
        return no
    elif nh is None:
        # Only iterations are in running
        return nr
    else:
        # Some iterations saved and some running
        return max(nr, nh) 
        
# Get the number of finished iterations
def GetHistoryIter():
    """Get the most recent iteration number for a history file
    
    This function uses the last line from the file ``run.resid``
    
    :Call:
        >>> n = pyOver.case.GetHistoryIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    # Read the project rootname
    try:
        rname = GetPrefix()
    except Exception:
        # Use "run" as prefix
        rname = "run"
    # Assemble file name.
    fname = "%s.resid" % rname
    # Check for the file.
    if not os.path.isfile(fname):
        # Alternative file
        fname = "%s.tail.resid" % rname
    # Check for the file.
    if not os.path.isfile(fname):
        # No history to read.
        return 0.0
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
        # Get the iteration number.
        return int(txt.split()[1])
    except Exception:
        # Failure; return no-iteration result.
        pass
        
# Get the last line (or two) from a running output file
def GetRunningIter():
    """Get the most recent iteration number for a running file
    
    This function uses the last line from the file ``resid.tmp``
    
    :Call:
        >>> n = pyOver.case.GetRunningIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2016-02-01 ``@ddalle``: First version
    """
    # Assemble file name.
    fname = "resid.tmp"
    # Check for the file.
    if not os.path.isfile(fname):
        # No history to read.
        return None
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
        # Get the iteration number.
        return int(txt.split()[1])
    except Exception:
        # Failure; return no-iteration result.
        return None
        
# Get the last line (or two) from a running output file
def GetOutIter():
    """Get the most recent iteration number for a running file
    
    This function uses the last line from the file ``resid.out``
    
    :Call:
        >>> n = pyOver.case.GetOutIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2016-02-02 ``@ddalle``: First version
    """
    # Assemble file name.
    fname = "resid.out"
    # Check for the file.
    if not os.path.isfile(fname):
        # No history to read.
        return None
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
        # Get the iteration number.
        return int(txt.split()[1])
    except Exception:
        # Failure; return no-iteration result.
        return None

# Function to get total iteration number
def GetRestartIter(rc=None):
    """Get total iteration number of most recent flow file
    
    :Call:
        >>> n = pyFun.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Get prefix
    rname = GetPrefix(rc)
    # Output glob
    fout = glob.glob('%s.[0-9][0-9]*.[0-9]*' % rname)
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fout:
        # Get the integer for this file.
        try:
            # Interpret the iteration number from file name
            i = int(fname.split('.')[-1])
        except Exception:
            # Failed to interpret this file name
            i = 0
        # Use the running maximum.
        n = max(i, n)
    # Output
    return n
    
# Function to set the most recent file as restart file.
def SetRestartIter(rc, n=None):
    """Set a given check file as the restart point
    
    :Call:
        >>> pyFun.case.SetRestartIter(rc, n=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2014-11-28 ``@ddalle``: Added `td_flowCart` compatibility
    """
    # Check the input.
    if n is None: n = GetRestartIter()
    # Read the namelist.
    nml = GetNamelist(rc)
    # Set restart flag
    if n > 0:
        # Set the restart flag on.
        nml.SetRestart()
    else:
        # Set the restart flag off.
        nml.SetRestart(False)
    # Write the namelist.
    nml.Write()
    # Get project name.
    fproj = GetProjectRootname()
    # Restart file name
    fname = '%s.flow' % fproj
    # Remove the current restart file if necessary.
    if os.path.islink(fname):
        # Remove the link
        os.remove(fname)
    elif os.path.isfile(fname):
        # Full file exists: abort!
        raise SystemError("Restart flow file '%s' already exists!" % fname)
    # Quit if no check point.
    if n == 0: return None
    # Source file
    fsrc = '%s.%i.flow' % (fproj, n)
    # Create a link to the most appropriate file.
    if os.path.isfile(fsrc):
        # Create the appropriate link.
        os.symlink(fsrc, fname)
        
# Check the number of iterations in an average
def checkqavg(fname):
    """Check the number of iterations in a ``q.avg`` file
    
    This function works by attempting to read a Fortran record at the very end
    of the file with exactly one (single-precision) integer. The function tries
    both little- and big-endian interpretations. If both methods fail, it
    returns ``1`` to indicate that the ``q`` file is a single-iteration
    solution.
    
    :Call:
        >>> nq = checkqavg(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of OVERFLOW ``q`` file
    :Outputs:
        *nq*: :class:`int`
            Number of iterations included in average
    :Versions:
        * 2016-12-29 ``@ddalle``: First version
    """
    # Open the file
    f = open(fname, 'rb')
    # Head to the end of the file, minus 12 bytes
    f.seek(-12, 2)
    # Try to read as a little-endian record at the end 
    I = np.fromfile(f, count=3, dtype="<i4")
    # If that failed to read 3 ints, file has < 12 bits
    if len(I) < 3:
        f.close()
        return 1
    # Check if the little-endian read came up with something
    if (I[0] == 4) and (I[2] == 4):
        f.close()
        return I[1]
    # Try a big-endian read
    f.seek(-12, 2)
    I = np.fromfile(f, count=3, dtype=">i4")
    f.close()
    # Check for success
    if (I[0] == 4) and (I[2] == 4):
        # This record makes sense
        return I[1]
    else:
        # Could not interpret record; assume one-iteration q-file
        return 1
        
# Check the iteration number 
def checkqt(fname):
    """Check the iteration number or time in a ``q`` file
    
    :Call:
        >>> t = checkqt(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of OVERFLOW ``q`` file
    :Outputs:
        *t*: ``None`` | :class:`float`
            Iteration number or time value
    :Versions:
        * 2016-12-29 ``@ddalle``: First version
    """
    # Open the file
    f = open(fname, 'rb')
    # Try to read the first record
    I = np.fromfile(f, count=1, dtype="<i4")
    # Check for valid read
    if len(I) == 0:
        f.close()
        return None
    # Check endianness
    if I[0] == 4:
        # Little endian
        ti = "<i4"
        tf = "<f"
    else:
        ti = ">i4"
        tf = ">f"
    # Read number of grids
    ng, i = np.fromfile(f, count=2, dtype=ti)
    # Check consistency
    if i != 4:
        f.close()
        return None
    # Read past the grid dimensions
    f.seek(4 + 12*ng, 1)
    # Read the number of states, num species, and end-of-record
    nq, nqc, i = np.fromfile(f, count=3, dtype=ti)
    # Read the header start-of-record marker to determine sp/dp
    i, = np.fromfile(f, count=1, dtype=ti)
    # Check for single precision
    if i == (13+max(2,nqc))*8 + 4:
        # Double-precision (usual)
        nf = 8
        tf = tf + "8"
    else:
        # Single-precision
        nf = 4
        tf = tf + "4"
    # Skip the first three entries of the header (REFMACH, ALPHA, REY)
    f.seek(3*nf, 1)
    # Read the time
    t, = np.fromfile(f, count=1, dtype=tf)
    # Close the file
    f.close()
    # Output
    return t
    
# Edit lines of a ``splitmq`` or ``splitmx`` input file
def EditSplitmqI(fin, fout, qin, qout):
    """Edit the I/O file names in a ``splitmq``/``splitmx`` input file
    
    :Call:
        >>> EditSplitmqI(fin, fout, qin, qout)
    :Inputs:
        *fin*: :class:`str`
            Name of template ``splitmq`` input file
        *fout*: :class:`str`
            Name of altered ``splitmq`` input file
        *qin*: :class:`str`
            Name of input solution or grid file
        *qout*: :class:`str`
            Name of output solution or grid file
    :Versions:
        * 2017-01-07 ``@ddalle``: First version
    """
    # Check for input file
    if not os.path.isfile(fin):
        raise ValueError("No template ``splitmq`` file '%s'" % fin)
    # Open the template and output files
    fi = open(fin, 'r')
    fo = open(fout, 'w')
    # Write the input and output solution/grid files
    fo.write('%s\n' % qin)
    fo.write('%s\n' % qout)
    # Ignore first two lines of input file
    fi.readline()
    fi.readline()
    # Copy the rest of the file
    fo.write(fi.read())
    # Close files
    fi.close()
    fo.close()
        
# Get best Q file
def GetQ():
    """Get the most recent ``q.*`` file, with ``q.avg`` taking precedence
    
    :Call:
        >>> fq = pyOver.case.GetQ()
    :Outputs:
        *fq*: ``None`` | :class:`str`
            Name of most recent averaged ``q`` file or most recent ``q`` file
    :Versions:
        * 2016-12-29 ``@ddalle``: First version
    """
    # Get the list of q files
    qglob = glob.glob('q.save')+glob.glob('q.restart')+glob.glob('q.[0-9]*')
    qavgb = glob.glob('q.avg*')
    # Check for averaged files
    if len(qavgb) > 0: qglob = qavgb
    # Exit if no files
    if len(qglob) == 0: return None
    # Get modification times from the files
    tq = [os.path.getmtime(fq) for fq in qglob]
    # Get index of most recent file
    iq = np.argmax(tq)
    # Return that file
    return qglob[iq]
    
# Get best q file
def GetLatest(glb):
    """Get the most recent file matching a glob or list of globs
    
    :Call:
        >>> fq = pyOver.case.GetLatest(glb)
        >>> fq = pyOver.case.GetLatest(lglb)
    :Inputs:
        *glb*: :class:`str`
            File name glob
        *lblb*: :class:`list` (:class:`str`)
            List of file name globs
    :Outputs:
        *fq*: ``None`` | :class:`str`
            Name of most recent file matching glob(s)
    :Versions:
        * 2017-01-08 ``@ddalle``: First version
    """
    # Check type
    if type(glb).__name__ in ['list', 'ndarray']:
        # Initialize from list of globs
        fglb = []
        # Loop through globs
        for g in glb:
            # Add the matches to this glob (don't worry about duplicates)
            fglb += glob.glob(g)
    else:
        # Single glob
        fglb = glob.glob(glb)
    # Exit if none
    if len(fglb) == 0: return None
    # Get modification times from the files
    tg = [os.path.getmtime(fg) for fg in fglb]
    # Get index of most cecent file
    ig = np.argmax(tg)
    # return that file
    return fglb[ig]
    
# Generic link command that cleans out existing links before making a mess
def LinkLatest(fsrc, fname):
    """Create a symbolic link, but clean up existing links
    
    This prevents odd behavior when using :func:`os.symlink` when the link
    already exists.  It performs no action (rather than raising an error) when
    the source file does not exist or is ``None``.  Finally, if *fname* is
    already a full file, no action is taken.
    
    :Call:
        >>> pyOver.case.LinkLatest(fsrc, fname)
    :Inputs:
        *fsrc*: ``None`` | :class:`str`
            Name of file to act as source for the link
        *fname*: :class:`str`
            Name of the link to create
    :Versions:
        * 2017-01-08 ``@ddalle``: First version
    """
    # Check for file
    if os.path.islink(fname):
        # Delete old links
        try:
            os.remove(fname)
        except Exception:
            pass
    elif os.path.isfile(fname):
        # Do nothing if full file exists with this name
        return
    # Check if the source file exists
    if (fsrc is None) or (not os.path.isfile(fsrc)): return
    # Create link
    try:
        os.symlink(fsrc, fname)
    except Exception:
        pass

# Link best Q file
def LinkQ():
    """Link the most recent ``q.*`` file to a fixed file name
    
    :Call:
        >>> pyOver.case.LinkQ()
    :Versions:
        * 2016-09-06 ``@ddalle``: First version
        * 2016-12-29 ``@ddalle``: Moved file search to :func:`GetQ`
    """
    # Get the general best ``q`` file name
    fq = GetQ()
    # Get the best single-iter, ``q.avg``, and ``q.srf`` files
    fqv = GetLatest(["q.[0-9]*[0-9]", "q.save", "q.restart"])
    fqa = GetLatest(["q.[0-9]*.avg", "q.avg*"])
    fqs = GetLatest(["q.[0-9]*.srf", "q.srf*", "q.[0-9]*.surf", "q.surf*"])
    # Create links (safely)
    LinkLatest(fq,  'q.pyover.p3d')
    LinkLatest(fqv, 'q.pyover.vol')
    LinkLatest(fqa, 'q.pyover.avg')
    LinkLatest(fqs, 'q.pyover.srf')
        
# Get best Q file
def GetX():
    """Get the most recent ``x.*`` file
    
    :Call:
        >>> fx = pyOver.case.GetX()
    :Outputs:
        *fx*: ``None`` | :class:`str`
            Name of most recent ``x.save`` or similar file
    :Versions:
        * 2016-12-29 ``@ddalle``: First version
    """
    # Get the list of q files
    xglob = (glob.glob('x.save') + glob.glob('x.restart') +
        glob.glob('x.[0-9]*') + glob.glob('grid.in'))
    # Exit if no files
    if len(xglob) == 0: return
    # Get modification times from the files
    tx = [os.path.getmtime(fx) for fx in xglob]
    # Get index of most recent file
    ix = np.argmax(tx)
    # Output
    return xglob[ix]
    
# Link best X file
def LinkX():
    """Link the most recent ``x.*`` file to a fixed file name
    
    :Call:
        >>> pyOver.case.LinkX()
    :Versions:
        * 2016-09-06 ``@ddalle``: First version
    """
    # Get the best file
    fx = GetX()
    # Get the best surf grid if available
    fxs = GetLatest(["x.[0-9]*.srf", "x.srf*", "x.[0-9]*.surf", "x.surf*"])
    # Create links (safely)
    LinkLatest(fx,  'x.pyover.p3d')
    LinkLatest(fxs, 'x.pyover.srf')
# def LinkX


# Function to determine newest triangulation file
def GetQFile(fqi="q.pyover.p3d"):
    """Get most recent OVERFLOW ``q`` file and its associated iterations
    
    Averaged solution files, such as ``q.avg`` take precedence.
    
    :Call:
        >>> fq, n, i0, i1 = GetQFile(fqi="q.pyover.p3d")
    :Inputs:
        *fqi*: {q.pyover.p3d} | q.pyover.avg | q.pyover.vol | :class:`str`
            Target Overflow solution file after linking most recent files
    :Outputs:
        *fq*: :class:`str`
            Name of ``q`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
        * 2017-03-28 ``@ddalle``: Moved from :mod:`lineLoad` to :mod:`case`
    """
    # Link grid and solution files
    LinkQ()
    LinkX()
    # Check for the input file
    if os.path.isfile(fqi):
        # Use the file (may be a link, in fact it usually is)
        fq = fqi
    else:
        # Best Q file available (usually "q.avg" or "q.save")
        fq = GetQ()
    # Check for q.avg iteration count
    n = checkqavg(fq)
    # Read the current "time" parameter
    i1 = checkqt(fq)
    # Get start parameter
    if (n is not None) and (i1 is not None):
        # Calculate start iteration
        i0 = i1 - n + 1
    else:
        # Cannot determine start iteration
        i0 = None
    # Output
    return fq, n, i0, i1
# def GetQFile
    

# Get initial settings
try:
    rc_init = ReadCaseJSON()
    j_init = GetPhaseNumber(rc_init)
    # Initial PBS script
    fpbs = "run_overflow.%02i.pbs" % j_init
    # Check if it exists.
    if not os.path.isfile(fpbs):
        # Single PBS script
        fpbs = "run_overflow.pbs"
    # Read it for wall time
    if os.path.isfile(fpbs):
        # Read for 'walltime'
        lines = bin.grep('walltime=', fpbs)
        # Read wall time
        txt_walltime = lines[0].split('=')[-1]
        # Convert to hr,min,seconds
        hrs, mins, secs = txt_walltime.split(':')
        # Convert to seconds
        twall_avail = 3600.0*int(hrs) + 60.0*int(mins) + 1.0*int(secs)
    else:
        # Available wall time unlimited
        twall_avail = 1e99
except Exception as e:
    # Unlimited wall time
    twall_avail = 1e99

