"""
Case control module: :mod:`pyFun.case`
======================================

This module contains functions to execute FUN3D and interact with individual
case folders.
"""

# Import cape stuff
from cape.case import *
import cape.manage as manage
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
        * 2016-04-05 ``@ddalle``: Added AFLR3 to this function
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
    # Determine the run index.
    i = GetPhaseNumber(rc)
    # Write the start time
    WriteStartTime(tic, rc, i)
    # Prepare files
    PrepareFiles(rc, i)
    # Prepare environment variables (other than OMP_NUM_THREADS)
    PrepareEnvironment(rc, i)
    # Run the appropriate commands
    RunPhase(rc, i)
    # Clean up files
    FinalizeFiles(rc, i)
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Save time usage
    WriteUserTime(tic, rc, i)
    # Check for errors
    CheckSuccess(rc, i)
    # Resubmit/restart if this point is reached.
    RestartCase(i)

# Prepare the files of the case
def PrepareFiles(rc, i=None):
    """Prepare file names appropriate to run phase *i* of FUN3D
    
    :Call:
        >>> PrepareFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
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
    # Check for dual phase
    if rc.get_Dual(): os.chdir('Flow')
    # Delete any input file.
    if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
        os.remove('fun3d.nml')
    # Create the correct namelist.
    os.symlink('fun3d.%02i.nml' % i, 'fun3d.nml')
    # Return to original folder
    if rc.get_Dual(): os.chdir('..')

# Run one phase appropriately
def RunPhase(rc, i):
    """Run one phase using appropriate commands
    
    :Call:
        >>> RunPhase(rc, i)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-04-13 ``@ddalle``: First version
    """
    # Count number of times this phase has been run previously.
    nprev = len(glob.glob('run.%02i.*' % i))
    # Check for dual
    if rc.get_Dual():
        os.chdir('Flow')
    # Read namelist
    nml = GetNamelist(rc, i)
    # Get the project name
    fproj = GetProjectRootname(rc=rc, i=i, nml=nml)
    # Get the last iteration number
    n = GetCurrentIter()
    # Number of requested iters for the end of this phase
    np = rc.get_PhaseIters(i)
    # Number of iterations to run this phase
    ni = rc.get_nIter(i)
    # Mesh generation and verification actions
    if i == 0 and n is None:
        # Run intersect and verify
        CaseIntersect(rc, fproj, n)
        CaseVerify(rc, fproj, n)
        # Create volume mesh if necessary
        CaseAFLR3(rc, proj=fproj, fmt=nml.GetGridFormat(), n=n)
        # Check for mesh-only phase
        if np is None or ni is None or ni <= 0 or np < 0:
            # Name of next phase
            fproj_adapt = GetProjectRootname(rc, i=i+1, nml=nml)
            # AFLR3 output format
            fmt = nml.GetGridFormat()
            # Check for renamed file
            if fproj_adapt != fproj:
                # Copy mesh
                os.symlink('%s.%s' % (fproj,fmt), '%s.%s' % (fproj_adapt,fmt))
            # Make sure *n* is not ``None``
            if n is None: n = 0
            # Exit appropriately
            if rc.get_Dual(): os.chdir('..')
            # Create an output file to make phase number programs work
            os.system('touch run.%02i.%i' % (i, n))
            return
    # Prepare for restart if that's appropriate.
    SetRestartIter(rc)
    # Check if the primal solution has already been run
    if n < np or nprev == 0:
        # Get the `nodet` or `nodet_mpi` command
        cmdi = cmd.nodet(rc)
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
    if (i>=rc.get_PhaseSequence(-1)) and (n>=rc.get_LastIter()):
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
        os.symlink('fun3d.dual.%02i.nml'%i, 'fun3d.nml')
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
        cmdi = cmd.nodet(rc, adapt=True)
        # Call the command.
        bin.callf(cmdi, f='adapt.out')
        # Rename output file after completing that command
        os.rename('adapt.out', 'adapt.%02i.out' % i)
        # Return home if appropriate
        if rc.get_Dual(): os.chdir('..')
        
# Check success
def CheckSuccess(rc=None, i=None):
    """Check for errors before continuing
    
    Currently the following checks are performed.
    
        * Check for NaN residual in the output file
        
    :Call:
        >>> CheckSuccess(rc=None, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Outputs:
        *q*: :class:`bool`
            Whether or not the case ran successfully
    :Versions:
        * 2016-04-18 ``@ddalle``: First version
    """
    # Get phase number if necessary.
    if i is None:
        # Get locally.
        i = GetPhaseNumber(rc)
    # Get the last iteration number
    n = GetCurrentIter()
    # Don't use ``None`` for this
    if n is None: n = 0
    # Output file name
    fname = 'run.%02i.%i' % (i, n)
    # Check for the file
    if os.path.isfile(fname):
        # Get the last line from nodet output file
        line = bin.tail(fname)
        # Check if NaN is in there
        if 'NaN' in line:
            raise RuntimeError("Found NaN locations!")

# Clean up immediately after running
def FinalizeFiles(rc, i=None):
    """Clean up files after running one cycle of phase *i*
    
    :Call:
        >>> FinalizeFiles(rc, i=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Options interface from ``case.json``
        *i*: :class:`int`
            Phase number
    :Versions:
        * 2016-04-14 ``@ddalle``: First version
    """
    # Get phase number if necessary.
    if i is None:
        # Get locally.
        i = GetPhaseNumber(rc)
    # Read namelist
    nml = GetNamelist(rc, i)
    # Get the project name
    fproj = GetProjectRootname(nml=nml)
    # Clean up the folder as appropriate.
    #manage.ManageFilesProgress(rc)
    # Get the last iteration number
    n = GetCurrentIter()
    # Don't use ``None`` for this
    if n is None: n = 0
    # Check for dual folder setup
    if os.path.isdir('Flow'):
        # Enter the flow folder
        os.chdir('Flow')
        qdual = True
        # History gets moved to parent
        fhist = os.path.join('..', 'run.%02i.%i' % (i,n))
    else:
        # Single folder
        qdual = False
        # History remains in present folder
        fhist = 'run.%02i.%i' % (i,n)
    # Assuming that worked, move the temp output file.
    if os.path.isfile('fun3d.out'):
        # Move the file
        os.rename('fun3d.out', fhist)
    else:
        # Create an empty file
        os.system('touch %s' % fhist)
    # Rename the flow file, too.
    if rc.get_KeepRestarts(i):
        shutil.copy('%s.flow' % fproj, '%s.%i.flow' % (fproj,n))
    # Move back to parent folder if appropriate
    if qdual: os.chdir('..')
        
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
    # Get restart iteration
    n = GetRestartIter()
    # Check for exit
    if n >= rc.get_LastIter():
        return
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
    
# Write start time
def WriteStartTime(tic, rc, i, fname="pyfun_start.dat"):
    """Write the start time in *tic*
    
    :Call:
        >>> WriteStartTime(tic, rc, i, fname="pyfun_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`pyOver.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: {``"pyfun_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2016-08-31 ``@ddalle``: First version
    """
    # Call the function from :mod:`cape.case`
    WriteStartTimeProg(tic, rc, i, fname, 'run_fun3d.py')
    
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
    # Global options
    qdual = rc.get_Dual()
    qadpt = rc.get_Adaptive()
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
        # Check for dual
        if qdual and rc.get_DualPhase(i):
            # Check for the dual output file
            if not os.path.isfile(os.path.join('Adjoint', 'dual.%02i.out'%i)):
                return i
        # Check for dual
        if qadpt and rc.get_AdaptPhase(i):
            # Check for weird hybrid setting
            if qdual:
                # It's in the ``Flow/`` folder; other phases may be dual phases
                fadpt = os.path.join('Flow', 'dual.%02i.out' % i)
            else:
                # Purely adaptive; located in this folder
                fadpt = 'dual.%02i.out' % i
            # Check for the dual output file
            if not os.path.isfile(fadpt):
                return i
    # Case completed; just return the last value.
    return i

# Get the namelist
def GetNamelist(rc=None, i=None):
    """Read case namelist file
    
    :Call:
        >>> nml = pyFun.case.GetNamelist(rc=None, i=None)
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
    # Read ``case.json`` if necessary
    if rc is None:
        try:
            rc = ReadCaseJSON()
        except Exception:
            pass
    # Process phase number
    if i is None and rc is not None:
        # Default to most recent phase number
        i = GetPhaseNumber(rc)
    # Check for `Flow` folder
    if os.path.isdir('Flow'):
        # Enter the folder
        qdual = True
        os.chdir('Flow')
    else:
        # No `Flow/` folder
        qdual = False
    # Check for folder with no working ``case.json``
    if rc is None:
        # Check for simplest namelist file
        if os.path.isfile('fun3d.nml'):
            # Read the currently linked namelist.
            nml = Namelist('fun3d.nml')
        else:
            # Look for namelist files
            fglob = glob.glob('fun3d.??.nml')
            # Sort it
            fglob.sort()
            # Read one of them.
            nml = Namelist(fglob[-1])
        # Return home if appropriate
        if qdual: os.chdir('..')
        return nml
    # Get the specified namelist
    nml = Namelist('fun3d.%02i.nml' % i)
    # Exit `Flow/` folder if necessary
    if qdual: os.chdir('..')
    # Output
    return nml


# Get the project rootname
def GetProjectRootname(rc=None, i=None, nml=None):
    """Read namelist and return project namelist
    
    :Call:
        >>> rname = pyFun.case.GetProjectRootname()
        >>> rname = pyFun.case.GetProjectRootname(rc=None, i=None, nml=None)
    :Inputs:
        *rc*: :class:`pyFun.options.runControl.RunControl`
            Run control options
        *i*: :class:`int`
            Phase number
        *nml*: :class:`pyFun.namelist.Namelist`
            Namelist interface; overrides *rc* and *i* if used
    :Outputs:
        *rname*: :class:`str`
            Project rootname
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
    """
    # Read a namelist.
    if nml is None: nml = GetNamelist(rc=rc, i=i)
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
        * 2016-04-28 ``@ddalle``: Accounting for ``Flow/`` folder
    """
    # Read the two sources
    nh, ns = GetHistoryIter()
    nr = GetRunningIter()
    # Process
    if nr is None:
        # No running iterations; check history
        return ns
    else:
        # Some iterations saved and some running
        return nh + nr
        
# Get the number of finished iterations
def GetHistoryIter():
    """Get the most recent iteration number for a history file
    
    :Call:
        >>> nh, n = pyFun.case.GetHistoryIter()
    :Outputs:
        *nh*: :class:`int`
            Iterations from previous cases before Fun3D deleted history
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2015-10-20 ``@ddalle``: First version
        * 2016-04-28 ``@ddalle``: Accounting for ``Flow/`` folder
        * 2016-10-29 ``@ddalle``: Handling Fun3D's iteration reset
    """
    # Check for flow folder
    if os.path.isdir('Flow'):
        # Dual setup
        qdual = True
        os.chdir('Flow')
    else:
        # No dual setup
        qdual = False
    # Read the project rootname
    try:
        rname = GetProjectRootname()
    except Exception:
        # No iterations
        if qdual: os.chdir('..')
        return None, None
    # Assemble file name.
    fname = "%s_hist.dat" % rname
    # Check for "pyfun00", "pyfun01", etc.
    if qdual:
        # Check for sequence of file names
        fnames = glob.glob(rname[:-2] + '??_hist.dat')
        # Sort descending
        fnames.sort()
        fnames.reverse()
    else:
        # Check for historical files
        fnames = glob.glob("%s_hist.[0-9][0-9].dat" % rname)
        fnames.sort()
        # Single history file name
        fnames.append("%s_hist.dat" % rname)
    # Loop through possible file(s)
    n = None
    nh = 0
    for fname in fnames:
        # Process the file
        ni = GetHistoryIterFile(fname)
        # Add to history
        if ni is not None:
            # Check if any iterations have been found
            if n is None:
                # First find
                n = ni
            else:
                # Add this history to previous history
                nh = n
                n += ni
    # No history to read.
    if qdual: os.chdir('..')
    # Output
    return nh, n
        
# Get the number of iterations from a single iterative history file
def GetHistoryIterFile(fname):
    """Get the most recent iteration number from a history file
    
    :Call:
        >>> n = pyFun.case.GetHistoryIterFile(fname)
    :Inputs:
        *fname*: {``"pyfun_hist.dat"``} | :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`int` | ``None``
            Most recent iteration number
    :Versions:
        * 2016-05-04 ``@ddalle``: Extracted from :func:`GetHistoryIter`
    """
    # Check for the file.
    if not os.path.isfile(fname):
        return None
    # Check the file.
    try:
        # Tail the file
        txt = bin.tail(fname)
    except Exception:
        # Failure; return no-iteration result.
        if qdual: os.chdir('..')
        return None
    # Get the iteration number.
    try:
        return int(txt.split()[0])
    except Exception:
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
        * 2016-04-28 ``@ddalle``: Now handles ``Flow/`` folder
    """
    # Check for the file.
    if os.path.isfile('fun3d.out'):
        # Use the current folder
        fflow = 'fun3d.out'
    elif os.path.isfile(os.path.join('Flow', 'fun3d.out')):
        # Use the ``Flow/`` folder
        fflow = os.path.join('Flow', 'fun3d.out')
    else:
        # No current file
        return None
    # Check for flag to ignore restart history
    lines = bin.grep('on_nohistorykept', fflow)
    # Check whether or not to add restart iterations
    if len(lines) < 2:
        # Get the restart iteration line
        try:
            # Search for particular text
            lines = bin.grep('the restart files contains', fflow)
            # Process iteration count from the RHS of the last such line
            nr = int(lines[0].split('=')[-1])
        except Exception:
            # No restart iterations
            nr = None
    else:
        # Do not use restart iterations
        nr = None
    # Get the last few lines of :file:`fun3d.out`
    lines = bin.tail(fflow, 100).strip().split('\n')
    lines.reverse()
    # Initialize output
    n = None
    # Try each line.
    for line in lines:
        try:
            # Check for direct specification
            if 'current history iterations' in line:
                # Direct specification
                n = int(line.split()[-1])
                nr = None
                break
            # Make sure there are iterations
            if len(line.split()) < 2: continue
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

    This function works by checking FUN3D output files for particular lines of
    text.  If the ``fun3d.out`` file exists, only that file is checked.
    Otherwise, all files matching ``run.[0-9]*.[0-9]*`` are checked.

    The lines in the FUN3D output file that report each new restart file have
    the following format.

        .. code-block:: none

            inserting previous and current history iterations 3300 + 800 = 4100
    
    :Call:
        >>> n = pyFun.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2015-10-19 ``@ddalle``: First version
        * 2016-04-19 ``@ddalle``: Checks STDIO file for iteration number
    """
    # List of saved run files
    frun = glob.glob('run.[0-9]*.[0-9]*')
    # Sort descending
    frun.sort()
    frun.reverse()
    # List the output files
    if os.path.isfile('fun3d.out'):
        # Only use the current file
        fflow = ['fun3d.out'] + frun
    elif os.path.isfile(os.path.join('Flow', 'fun3d.out')):
        # Use the current file from the ``Flow/`` folder
        fflow = [os.path.join('Flow', 'fun3d.out')] + frun
    else:
        # Use the run output files
        fflow = frun
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fflow:
        # Get the output report lines
        lines = bin.grep('current history iterations', fname)
        # Be safe
        try:
            # Get the last write iteration number
            i = int(lines[-2].split()[-1])
            # Update iteration number
            n = max(i, n)
        except Exception:
            pass
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
        # Get the phase
        i = GetPhaseNumber(rc)
        # Check if this is a phase restart
        nohist = True
        if os.path.isfile('run.%02i.%i' % (i, n)):
            # Nominal restart
            nohist = False
        elif i == 0:
            # Not sure how we could still be in phase 0
            nohist = False
        else:
            # Read the previous namelist
            nml0 = GetNamelist(rc, i-1)
            # Get 'time_accuracy' parameter
            ta0 = nml0.GetVar('nonlinear_solver_parameters', 'time_accuracy')
            ta1 = nml.GetVar( 'nonlinear_solver_parameters', 'time_accuracy')
            # Check for a match
            nohist = ta0 != ta1
            # If we are moving to a new mode, prevent Fun3D deleting history
            if nohist:
                CopyHist(nml0, i-1)
        # Set the restart flag on.
        nml.SetRestart(nohist=nohist)
    else:
        # Set the restart flag off.
        nml.SetRestart(False)
    # Write the namelist.
    nml.Write()
    
# Copy the histories
def CopyHist(nml, i):
    """Copy all force and moment histories along with residual history
    
    :Call:
        >>> CopyHist(nml, i)
    :Inputs:
        *nml*: :class:`pyFun.namelist.Namelist`
            Fun3D namelist interface for phase *i*
        *i*: :class:`int`
            Phase number to use for storing histories
    :Versions:
        * 2016-10-28 ``@ddalle``: First version
    """
    # Project name
    proj = nml.GetRootname()
    # Get the list of FM files
    fmglob = glob.glob('%s_fm_*.dat' % proj)
    # Loop through FM files
    for f in fmglob:
        # Split words
        F = f.split('.')
        # Copy-to name
        fcopy = '.'.join(F[:-1]) + ('.%02i.dat' % i)
        # Copy the file
        os.rename(f, fcopy)
    # Copy the history file
    if os.path.isfile('%s_hist.dat' % proj):
        # Copy the file
        os.rename(
            '%s_hist.dat' % proj,
            '%s_hist.%02i.dat' % (proj, i))
    # Copy the history file
    if os.path.isfile('%s_subhist.dat' % proj):
        # Copy the file
        os.rename(
            '%s_subhist.dat' % proj,
            '%s_subhist.%02i.dat' % (proj, i))
    
    
    
# Get best file based on glob
def GetFromGlob(fglb):
    """Find the most recently edited file matching a glob
    
    :Call:
        >>> fname = pyFun.case.GetFromGlob(fglb)
    :Inputs:
        *fglb*: :class:`str`
            Glob for targeted file names
    :Outputs:
        *fname*: :class:`str`
            Name of file matching glob that was most recently edited
    :Versions:
        * 2016-12-19 ``@ddalle``: First version
    """
    # List of files matching requested glob
    fglob = glob.glob(fglb)
    # File extension
    fext = '.' + fglb.split('.')[-1]
    # Check for empty glob
    if len(fglob) == 0: return
    # Get modification times
    t = [os.path.getmtime(f) for f in fglob]
    # Extract file with maximum index
    fname = fglob[t.index(max(t))]
    # Output
    return fname
    
# Link best file based on name and glob
def LinkFromGlob(fname, fglb):
    """Link the most recent file to a generic Tecplot file name
    
    :Call:
        >>> pyFun.case.LinkFromGlob(fname, fglb)
    :Inputs:
        *fname*: :class:`str`
            Name of unmarked file, like ``Components.i.plt``
        *fglb*: :class:`str`
            Glob for marked file names
    :Versions:
        * 2016-10-24 ``@ddalle``: First version
    """
    # Check for already-existing regular file
    if os.path.isfile(fname) and not os.path.islink(fname): return
    # Remove the link if necessary
    if os.path.isfile(fname) or os.path.islink(fname):
        os.remove(fname)
    # Extract file with maximum index
    fsrc = GetFromGlob(fglb)
    # Create the link if possible
    if os.path.isfile(fsrc): os.symlink(fsrc, fname)
    
# Link best Tecplot files
def LinkPLT():
    """Link the most recent Tecplot files to fixed file names
    
    :Call:
        >>> pyFun.case.LinkPLT()
    :Versions:
        * 2016-10-24 ``@ddalle``: First version
    """
    # Read the options
    rc = ReadCaseJSON()
    j = GetPhaseNumber(rc)
    # Need the namelist to figure out planes, etc.
    nml = GetNamelist(rc=rc, i=j)
    # Get the project root name
    proj = nml.GetVar('project', 'project_rootname')
    # Strip suffix
    if rc.get_Dual(j):
        # Strip adaptive section
        proj0 = proj[:-2]
    else:
        # Use the full project name if no adaptations
        proj0 = proj
    # Get the list of output surfaces
    fsrf = []
    i = 1
    flbl = nml.GetVar('sampling_parameters', 'label', i)
    # Loop until there's no output surface name
    while flbl is not None:
        # Append
        fsrf.append(flbl)
        # Move to sampling output *i*
        i += 1
        # Get the name
        flbl = nml.GetVar('sampling_parameters', 'label', i)
    # Initialize file names
    fname = [
        '%s_tec_boundary' % proj0,
        '%s_tec_volume' % proj0,
    ]
    # Initialize globs
    fglob = [
        '%s_tec_boundary_timestep*' % proj,
        '%s_tec_volume_timestep*' % proj
    ]
    # Add special ones
    for fi in fsrf:
        fname.append('%s_%s' % (proj0, fi))
        fglob.append('%s_%s_timestep*' % (proj0, fi))
    # Link the globs
    for i in range(len(fname)):
        # Process the glob as well as possible
        LinkFromGlob(fname[i]+".dat", fglob[i]+".dat")
        LinkFromGlob(fname[i]+".plt", fglob[i]+".plt")
    
    
# def LinkPLT

