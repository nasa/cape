"""
Case control module: :mod:`pyFun.case`
======================================

This module contains functions to execute FUN3D and interact with individual
case folders.
"""

# Import cape stuff
from cape.case import *
# Import options class
from options.runControl import RunControl
# Import the namelist
from namelist import Namelist
# Interface for writing commands
from . import bin, cmd, queue


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
        raise IOError('Case already running!')
    # Touch the running file.
    os.system('touch RUNNING')
    # Start timer
    tic = datetime.now()
    # Get the run control settings
    rc = ReadCaseJSON()
    # Get the project name
    fproj = GetProjectRootname()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Delete any input file.
    if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
        os.remove('fun3d.nml')
    # Create the correct namelist.
    os.symlink('fun3d.%02i.nml'%i, 'fun3d.nml')
    # Prepare for restart if that's appropriate.
    SetRestartIter(rc)
    # Get the `nodet` or `nodet_mpi` command
    cmdi = cmd.nodet(rc)
    # Call the command.
    bin.callf(cmdi, f='fun3d.out')
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Save time usage
    WriteUserTime(tic, rc, i)
    # Get the last iteration number
    n = GetCurrentIter()
    # Assuming that worked, move the temp output file.
    os.rename('fun3d.out', 'run.%02i.%i' % (i, n))
    # Rename the flow file, too.
    if rc.get_KeepRestarts(i):
        shutil.copy('%s.flow' % fproj, '%s.%i.flow' % (fproj,n))
    # Check current iteration count.
    if (i>=rc.get_PhaseSequence(-1)) and (n>=rc.get_LastIter()):
        return
    # Check for next phase
    i1 = GetPhaseNumber(rc)
    # Check for adaptive solves
    if i1>i and rc.get_Adaptive() and rc.get_AdaptPhase(i):
        # Check for adjoint solver
        if rc.get_Dual(i):
            pass
        else:
            # Run the feature-based adaptive mesher
            cmdi = cmd.nodet(rc, adapt=True)
            # Call the command.
            bin.callf(cmdi, f='fun3d.out')
    # Resubmit/restart if this point is reached.
    RestartCase(i)

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
    i = GetPhaseNumber(rc)
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
        
# Function to call script or submit.
def RestartCase(i0=None):
    """Restart a case by either submitting it or calling with a system command
    
    This version of the command is called within :func:`run_fun3d` after
    running a phase or attempting to run a phase.
    
    :Call:
        >>> pyFun.case.RestartCase(i0=None)
    :Inputs:
        *i0*: :class:`int` | ``None``
            Run sequence index of the previous run
    :Versions:
        * 2015-12-30 ``@ddalle``: Split from pyCart
    """
    # Get the config.
    rc = ReadCaseJSON()
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Check qsub status.
    if not rc.get_qsub(i):
        # Run the case.
        run_fun3d()
    elif rc.get_Resubmit(i):
        # Check for continuance
        if (i0 is None) or (i>i0) or (not rc.get_Continue(i)):
            # Get the name of the PBS file.
            fpbs = GetPBSScript(i)
            # Submit the case.
            pbs = queue.pqsub(fpbs)
            return pbs
        else:
            # Continue on the same job
            run_fun3d()
    else:
        # Simply run the case. Don't reset modules either.
        run_fun3d()
    
    
# Write time used
def WriteUserTime(tic, rc, i, fname="pyfun_time.dat"):
    """Write time usage since time *tic* to file
    
    :Call:
        >>> toc = WriteUserTime(tic, rc, i, fname="pyfun_time.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyCart.options.runControl.RunControl
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
    """
    # Call the function from :mode:`cape.case`
    WriteUserTimeProg(tic, rc, i, fname, 'run_fun3d.py')

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
def GetPhaseNumber(rc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyFun.case.GetPhaseNumber(rc)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface for run control
    :Outputs:
        *i*: :class:`int`
            Most appropriate phase number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2015-10-19 ``@ddalle``: FUN3D version
    """
    # Get the run index.
    n = GetRestartIter()
    # Loop through possible input numbers.
    for j in range(rc.get_nSeq()):
        # Get the actual run number
        i = rc.get_PhaseSequence(j)
        # Check for output files.
        if len(glob.glob('run.%02i.*' % i)) == 0:
            # This run has not been completed yet.
            return i
        # Check the iteration number.
        if n < rc.get_PhaseIters(j):
            # This case has been run, but hasn't reached the min iter cutoff
            return i
    # Case completed; just return the last value.
    return i

# Get the namelist
def GetNamelist(rc=None, i=None):
    """Read case namelist file
    
    :Call:
        >>> nml = pyFun.case.GetNamelist(rc=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *i*: :class:`int`
            Phase number
    :Outputs:
        *nml*: :class:`pyFun.namelist.Namelist`
            Namelist interface
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Check for detailed inputs
    if i is not None:
        # Get the specified namelist
        return Namelist('fun3d.%02i.nml' % i)
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
        i = GetPhaseNumber(rc)
        # Read the namelist file.
        return Namelist('fun3d.%02i.nml' % i)


# Get the project rootname
def GetProjectRootname(rc=None, i=None):
    """Read namelist and return project namelist
    
    :Call:
        >>> rname = pyFun.case.GetProjectRootname()
        >>> rname = pyFun.case.GetProjectRootname(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *i*: :class:`int`
            Phase number
    :Outputs:
        *rname*: :class:`str`
            Project rootname
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read a namelist.
    nml = GetNamelist(rc=rc, i=i)
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
    else:
        # Some iterations saved and some running
        return nr
        
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
    # Get the restart iteration line
    try:
        # Search for particular text
        lines = bin.grep('the restart files contains', 'fun3d.out')
        # Process iteration count from the RHS of the last such line
        nr = int(lines[0].split('=')[-1])
    except Exception:
        # No restart iterations
        nr = None
    # Get the last few lines of :file:`fun3d.out`
    lines = bin.tail('fun3d.out', 7).strip().split('\n')
    lines.reverse()
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
    if n is None:
        return nr
    elif nr is None:
        return n
    else:
        return n + nr

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
    fflow = glob.glob('run.[0-9]*.[0-9]*')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fflow:
        # Get the integer for this file.
        i = int(fname.split('.')[-1])
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

