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


# Function to complete final setup and call the appropriate FUN3D commands
def run_fun3d():
    """Setup and run the appropriate FUN3D command
    
    :Call:
        >>> pyFun.case.run_fun3d()
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Check for RUNNING file.
    if os.path.isfile('RUNNING'):
        # Case already running
        raise IOError('Fase already running!')
    # Touch the running file.
    os.system('touch RUNNING')
    # Get the run control settings
    rc = ReadCaseJSON()
    # Get the project name
    fproj = GetProjectRootname()
    # Determine the run index.
    i = GetInputNumber(rc)
    # Set the restart file.
    SetRestartIter()
    # Delete any input file.
    if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
        os.remove('fun3d.nml')
    # Create the correct namelist.
    os.symlink('fun3d.%02i.nml'%i, 'fun3d.nml')
    # Check for `nodet` vs `nodet_mpi`
    if rc.get_MPI(i):
        # Get number of nodes
        nProc = str(rc.get_nProc(i))
        # Get the command to run MPI on this machine
        mpicmd = rc.get_mpicmd()
        # nodet_mpi command
        cmdi = [mpicmd, '-np', nProc, 'nodet_mpi', '--animation_freq', '-1']
    else:
        # nodet command
        cmdi = ['nodet', '--animation_freq', '-1']
    # Call the command.
    bin.callf(cmdi, f='fun3d.out')
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Get the last iteration number
    n = GetCurrentIter()
    # Assuming that worked, move the temp output file.
    os.rename('fun3d.out', 'run.%02i.%i' % (i, n))
    # Rename the flow file, too.
    os.rename('%s.flow' % fproj, '%s.%i.flow' % (fproj,n))
    # Check current iteration count.
    if n >= rc.get_LastIter():
        return
    # Resubmit if asked.
    if rc.get_resub(i):
        # Run full restart command, including qsub if appropriate
        StartCase()
    else:
        # Get the name of the PBS script
        fpbs = GetPBSScript(i)
        # Just run the case directly (keep the same PBS job).
        bin.callf(['bash', fpbs])

# Function to call script or submit.
def StartCase():
    """Start a case by either submitting it or calling with a system command
    
    :Call:
        >>> pyFun.case.StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: Copied from pyCart
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetInputNumber(rc)
    # Check qsub status.
    if rc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_fun3d()

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
    
# Function to chose the correct input to use from the sequence.
def GetInputNumber(rc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyFun.case.GetInputNumber(rc)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control
    :Outputs:
        *i*: :class:`int`
            Most appropriate run number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
    """
    # Get the run index.
    n = GetRestartIter()
    # Loop through possible input numbers.
    for j in range(rc.get_nSeq()):
        # Get the actual run number
        i = rc.get_InputSeq(j)
        # Check for output files.
        if len(glob.glob('run.%02i.*' % i)) == 0:
            # This run has not been completed yet.
            return i
        # Check the iteration number.
        if n < rc.get_IterSeq(j):
            # This case has been run, but hasn't reached the min iter cutoff
            return i
    # Case completed; just return the last value.
    return i

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
    # Check for detailed inputs
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
        # Get run index.
        i = GetInputNumber(rc)
        # Read the namelist file.
        return Namelist('fun3d.%s.nml' % i)


# Get the project rootname
def GetProjectRootname(rc=None):
    """Read namelist and return project namelist
    
    :Call:
        >>> rname = pyFun.case.GetProjectRootname()
        >>> rname = pyFun.case.GetProjectRootname(rc)
    :Outputs:
        *rname*: :class:`str`
            Project rootname
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read a namelist.
    nml = GetNamelist(rc=rc)
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
        *n*: :class:`int` | ``None``
            Last iteration number
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read the two sources
    nh = GetHistoryIter()
    nr = GetRunningIter()
    # Process
    if nr is None:
        # No running iterations; check history
        return nh
    elif nh is None:
        # No completed sets but some iterations running
        return nr
    else:
        # Some iterations saved and some running
        return nh + nr
        
# Get the number of finished iterations
def GetHistoryIter():
    """Get the most recent iteration number for a history file
    
    :Call:
        >>> n = pyFun.case.GetHistoryIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2015-10-20 ``@ddalle``: First version
    """
    # Read the project rootname
    try:
        rname = GetProjectRootname()
    except Exception:
        # No iterations
        return None
    # Assemble file name.
    fname = "%s_hist.dat" % rname
    # Check for the file.
    if not os.path.isfile(fname):
        # No history to read.
        return None
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
        # Get the iteration number.
        return int(txt.split()[0])
    except Exception:
        # Failure; return no-iteration result.
        return None
        
# Get the last line (or two) from a running output file
def GetRunningIter():
    """Get the most recent iteration number for a running file
    
    :Call:
        >>> n = pyFun.case.GetRunningIter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Check for the file.
    if not os.path.isfile('fun3d.out'): return None
    # Get the last three lines of :file:`fun3d.out`
    lines = bin.tail('fun3d.out', 3).strip().split('\n')
    # Initialize output
    n = None
    # Try each line.
    for line in lines:
        try:
            # Try to use an integer for the first entry.
            n = int(line.split()[0])
            break
        except Exception:
            continue
    # Output
    return n

# Function to get total iteration number
def GetRestartIter():
    """Get total iteration number of most recent flow file
    
    :Call:
        >>> n = pyFun.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # List the *.*.flow files
    fflow = glob.glob('*.[0-9][0-9]*.flow')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fflow:
        # Get the integer for this file.
        i = int(fname.split('.')[-2])
        # Use the running maximum.
        n = max(i, n)
    # Output
    return n
    
# Function to set the most recent file as restart file.
def SetRestartIter(n=None):
    """Set a given check file as the restart point
    
    :Call:
        >>> pyCart.case.SetRestartIter(n=None, ntd=None)
    :Inputs:
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2014-11-28 ``@ddalle``: Added `td_flowCart` compatibility
    """
    # Check the input.
    if n is None: n = GetRestartIter()
    # Get project name.
    fproj = GetProjectRootname()
    # Restart file name
    fname = '%s.flow' % fproj
    # Remove the current restart file if necessary.
    if os.path.isfile(fname):
        # Full file exists: abort!
        raise SystemError("Restart flow file '%s' already exists!" % fname)
    elif os.path.islink(fname):
        # Remove the link
        os.remove(fname)
    # Quit if no check point.
    if n == 0: return None
    # Source file
    fsrc = '%s.%i.flow' % (fproj, n)
    # Create a link to the most appropriate file.
    if os.path.isfile(fsrc):
        # Create the appropriate link.
        os.symlink(fsrc, fname)

