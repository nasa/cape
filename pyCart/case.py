"""
Case for interacting with individual run cases: :mod:`pyCart.case`
==================================================================

"""

# Import options class
from options.flowCart import flowCart
# Binary interface.
from bin import callf, tail
# Interface for writing commands
from . import cmd, queue, manage

# Read the local JSON file.
import json
# File control
import os, glob, shutil
# Basic numerics
from numpy import nan, isnan


# Function to setup and call the appropriate flowCart file.
def run_flowCart():
    """
    Setup and run the appropriate `flowCart`, `mpi_flowCart` command
    
    :Call:
        >>> pyCart.case.run_flowCart()
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2014-12-18 ``@ddalle``: Added :func:`TarAdapt` call after run
    """
    # Check for RUNNING file.
    if os.path.isfile('RUNNING'):
        # Case already running
        raise IOError('Case already running!')
    # Touch the running file.
    os.system('touch RUNNING')
    # Get the settings.
    fc = ReadCaseJSON()
    # Determine the run index.
    i = GetInputNumber(fc)
    # Create a restart file if appropriate.
    if not fc.get_use_aero_csh(i):
        # Automatically determine the best check file to use.
        SetRestartIter()
    # Delete any input file.
    if os.path.isfile('input.cntl') or os.path.islink('input.cntl'):
        os.remove('input.cntl')
    # Create the correct input file.
    os.symlink('input.%02i.cntl' % i, 'input.cntl')
    # Extra prep for adaptive --> non-adaptive
    if (i>0) and (not fc.get_use_aero_csh(i)) and (os.path.isdir('BEST')
            and (not os.path.isfile('history.dat'))):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Find all *.dat files and Mesh files
        fglob = glob.glob('*.dat') + glob.glob('Mesh.*')
        # Go back up one folder.
        os.chdir('..')
        # Copy all the important files.
        for fname in fglob:
            # Check for the file.
            if os.path.isfile(fname): continue
            # Copy the file.
            shutil.copy('BEST/'+fname, fname)
    # Convince aero.csh to use the *new* input.cntl
    if (i>0) and (fc.get_use_aero_csh(i)) and (fc.get_use_aero_csh(i-1)):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Check for an input.cntl file
        if os.path.isfile('input.cntl'):
            # Move it to a representative name.
            os.rename('input.cntl', 'input.%02i.cntl' % (i-1))
        # Go back up.
        os.chdir('..')
        # Copy the new input file.
        shutil.copy('input.%02i.cntl' % i, 'BEST/input.cntl')
    # Check for flowCart vs. mpi_flowCart
    if not fc.get_mpi_fc(i):
        # Get the number of threads, which may be irrelevant.
        nProc = fc.get_nProc()
        # Set it.
        os.environ['OMP_NUM_THREADS'] = str(nProc)
    # Check for adaptive runs.
    if fc.get_use_aero_csh(i):
        # Delete the existing aero.csh file
        if os.path.isfile('aero.csh'): os.remove('aero.csh')
        # Create a link to this run.
        os.symlink('aero.%02i.csh' % i, 'aero.csh')
        # Call the aero.csh command
        if i > 0 or GetCurrentIter() > 0:
            # Restart case.
            cmdi = ['./aero.csh', 'restart']
        elif fc.get_jumpstart():
            # Initial case
            cmdi = ['./aero.csh', 'jumpstart']
        else:
            # Initial case and create grid
            cmdi = ['./aero.csh']
    else:
        # Check how many iterations by which to offset the count.
        if fc.get_unsteady(i):
            # Get the number of previous unsteady steps.
            n = GetUnsteadyIter()
        else:
            # Get the number of previous steady steps.
            n = GetSteadyIter()
        # Call flowCart directly.
        cmdi = cmd.flowCart(fc=fc, i=i, n=n)
    # Run the command.
    callf(cmdi, f='flowCart.out')
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Clean up the folder as appropriate.
    # Tar visualization files.
    if fc.get_unsteady(i):
        manage.TarViz()
    # Tar old adaptation folders.
    if fc.get_use_aero_csh(i):
        manage.TarAdapt()
    # Last reported iteration number
    n = GetHistoryIter()
    # Check status
    if n % 1 != 0:
        # Ended with a failed unsteady cycle!
        f = open('FAIL', 'w')
        # Write the failure type.
        f.write('# Ended with failed unsteady cycle at iteration:\n')
        f.write('%13.6f\n' % n)
        # Quit
        f.close()
        return
    # Last reported residual
    L1 = GetCurrentResid()
    # Check for bad (large or NaN) values.
    if isnan(L1) or L1>1.0e+8:
        # Exploded.
        f = open('FAIL', 'w')
        # Write the failure type.
        f.write('# Bombed at iteration %.6f with residual %.2E.\n' % (n, L1))
        f.write('%13.6f\n' % n)
        # Quit
        f.close()
        return
    # Check for a hard-to-detect failure present in the output file.
    if CheckFailed():
        # Some other failure
        f = open('FAIL', 'w')
        # Copy the last line of flowCart.out
        f.write('# %s' % tail('flowCart.out'))
        # Quit
        f.close()
        return
    # Get the new restart iteration.
    n = GetCheckResubIter()
    # Assuming that worked, move the temp output file.
    os.rename('flowCart.out', 'run.%02i.%i' % (i, n))
    # Check for TecPlot files to save.
    if os.path.isfile('cutPlanes.plt'):
        os.rename('cutPlanes.plt', 'cutPlanes.%05i.plt' % n)
    if os.path.isfile('Components.i.plt'):
        os.rename('Components.i.plt', 'Components.%05i.plt' % n)
    # Clear check files as appropriate.
    manage.ClearCheck()
    # Check current iteration count.
    if n >= fc.get_LastIter():
        return
    # Resubmit if asked.
    if fc.get_resub(i):
        # Run full restart command, including qsub if appropriate
        StartCase()
    else:
        # Get the name of the PBS script
        fpbs = GetPBSScript(i)
        # Just run the case directly (keep the same PBS job).
        callf(['bash', fpbs])
    
    
# Function to call script or submit.
def StartCase():
    """Start a case by either submitting it or calling with a system command
    
    :Call:
        >>> pyCart.case.StartCase()
    :Versions:
        * 2014-10-06 ``@ddalle``: First version
    """
    # Get the config.
    fc = ReadCaseJSON()
    # Determine the run index.
    i = GetInputNumber(fc)
    # Check qsub status.
    if fc.get_qsub(i):
        # Get the name of the PBS file.
        fpbs = GetPBSScript(i)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_flowCart()
        
# Function to delete job and remove running file.
def StopCase():
    """Stop a case by deleting its PBS job and removing :file:`RUNNING` file
    
    :Call:
        >>> pyCart.case.StopCase()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    # Get the job number.
    jobID = queue.pqjob()
    # Try to delete it.
    queue.qdel(jobID)
    # Check if the RUNNING file exists.
    if os.path.isfile('RUNNING'):
        # Delete it.
        os.remove('RUNNING')
        
# Function to check output file for some kind of failure.
def CheckFailed():
    """Check the :file:`flowCart.out` file for a failure
    
    :Call:
        >>> q = pyCart.case.CheckFailed()
    :Outputs:
        *q*: :class:`bool`
            Whether or not the last line of `flowCart.out` contains 'fail'
    :Versions:
        * 2015-01-02 ``@ddalle``: First version
    """
    # Check for the file.
    if os.path.isfile('flowCart.out'):
        # Read the last line.
        if 'fail' in tail('flowCart.out', 1):
            # This is a failure.
            return True
        else:
            # Normal completed run.
            return False
    else:
        # No flowCart.out file
        return False

# Function to determine which PBS script to call
def GetPBSScript(i=None):
    """Determine the file name of the PBS script to call
    
    This is a compatibility function for cases that do or do not have multiple
    PBS scripts in a single run directory
    
    :Call:
        >>> fpbs = pyCart.case.GetPBSScript(i=None)
    :Inputs:
        *i*: :class:`int`
            Run index
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: First version
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if i is not None:
        # Create the name.
        fpbs = 'run_cart3d.%02i.pbs' % i
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return 'run_cart3d.pbs'
    else:
        # Do not search for numbered PBS script if *i* is None
        return 'run_cart3d.pbs'
    

# Function to read the local settings file.
def ReadCaseJSON():
    """Read `flowCart` settings for local case
    
    :Call:
        >>> fc = pyCart.case.ReadCaseJSON()
    :Outputs:
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Options interface for `flowCart`
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
    """
    # Read the file, fail if not present.
    f = open('case.json')
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a flowCart object.
    fc = flowCart(**opts)
    # Output
    return fc
    

# Function to get the most recent check file.
def GetSteadyIter():
    """Get iteration number of most recent steady check file
    
    :Call:
        >>> n = pyCart.case.GetSteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2014-11-28 ``@ddalle``: Renamed from :func:`GetRestartIter`
    """
    # List the check.* files.
    fch = glob.glob('check.*[0-9]') + glob.glob('BEST/check.*')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Loop through the matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname.split('.')[-1])
        # Use the running maximum.
        n = max(i, n)
    # Output
    return n
    
# Function to get the most recent time-domain check file.
def GetUnsteadyIter():
    """Get iteration number of most recent unsteady check file
    
    :Call:
        >>> n = pyCart.case.GetUnsteadyIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: First version
    """
    # Check for td checkpoints
    fch = glob.glob('check.*.td')
    # Initialize unsteady count
    n = 0
    # Loop through matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname.split('.')[1])
        # Use the running maximum.
        n = max(i, n)
    # Output.
    return n
    
# Function to get total iteration number
def GetRestartIter():
    """Get total iteration number of most recent check file
    
    This is the sum of the most recent steady iteration and unsteady iteration.
    
    :Call:
        >>> n = pyCart.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: First version
    """
    # Get the unsteady iteration number based on available check files.
    ntd = GetUnsteadyIter()
    # Check for an unsteady iteration number.
    if ntd:
        # If there's an unsteady iteration, use that step directly.
        return ntd
    else:
        # Use the steady-state iteration number.
        return GetSteadyIter()
    
# Function to get total iteration number
def GetCheckResubIter():
    """Get total iteration number of most recent check file
    
    This is the sum of the most recent steady iteration and unsteady iteration.
    
    :Call:
        >>> n = pyCart.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014-11-28 ``@ddalle``: First version
        * 2014-11-29 ``@ddalle``: This was renamed from :func:`GetRestartIter`
    """
    # Get the two numbers
    nfc = GetSteadyIter()
    ntd = GetUnsteadyIter()
    # Output
    return nfc + ntd
    
    
# Function to set up most recent check file as restart.
def SetRestartIter(n=None, ntd=None):
    """Set a given check file as the restart point
    
    :Call:
        >>> pyCart.case.SetRestartIter(n=None, ntd=None)
    :Inputs:
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
        *ntd*: :class:`int`
            Unsteady iteration number
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
        * 2014-11-28 ``@ddalle``: Added `td_flowCart` compatibility
    """
    # Check the input.
    if n   is None: n = GetSteadyIter()
    if ntd is None: ntd = GetUnsteadyIter()
    # Remove the current restart file if necessary.
    if os.path.isfile('Restart.file') or os.path.islink('Restart.file'):
        os.remove('Restart.file')
    # Quit if no check point.
    if n == 0 and ntd == 0:
        return None
    # Create a link to the most appropriate file.
    if os.path.isfile('check.%06i.td' % ntd):
        # Restart from time-accurate check point
        os.symlink('check.%06i.td' % ntd, 'Restart.file')
    elif os.path.isfile('BEST/check.%05i' % n):
        # Restart file in adaptive folder
        os.symlink('BEST/check.%05i' % n, 'Restart.file')
    elif os.path.isfile('check.%05i' % n):
        # Restart file in current folder
        os.symlink('check.%05i' % n, 'Restart.file')
    
    
# Function to chose the correct input to use from the sequence.
def GetInputNumber(fc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyCart.case.GetInputNumber(fc)
    :Inputs:
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Options interface for `flowCart`
    :Outputs:
        *i*: :class:`int`
            Most appropriate run number for a restart
    :Versions:
        * 2014-10-02 ``@ddalle``: First version
    """
    # Get the run index.
    n = GetCheckResubIter()
    # Loop through possible input numbers.
    for j in range(fc.get_nSeq()):
        # Get the actual run number
        i = fc.get_InputSeq(j)
        # Check for output files.
        if not glob.glob('run.%02i.*' % i):
            # This run has not been completed yet.
            return i
        # Check the iteration number.
        if n < fc.get_IterSeq(j):
            # This case has been run, but hasn't reached the min iter cutoff
            return i
    # Case completed; just return the last value.
    return i
    
# Function to read last line of 'history.dat' file
def GetHistoryIter(fname='history.dat'):
    """Get the most recent iteration number from a :file:`history.dat` file
    
    :Call:
        >>> n = pyCart.case.GetHistoryIter(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`float`
            Last iteration number
    :Versions:
        * 2014-11-24 ``@ddalle``: First version
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return 0
    # Check the file.
    try:
        # Try to tail the last line.
        txt = tail(fname)
        # Try to get the integer.
        return float(txt.split()[0])
    except Exception:
        # If any of that fails, return 0
        return 0
        
# Get last residual from 'history.dat' file
def GetHistoryResid(fname='history.dat'):
    """Get the last residual in a :file:`history.dat` file
    
    :Call:
        >>> L1 = pyCart.case.GetHistoryResid(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: First version
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return nan
    # Check the file.
    try:
        # Try to tail the last line.
        txt = tail(fname)
        # Try to get the integer.
        return float(txt.split()[3])
    except Exception:
        # If any of that fails, return 0
        return nan

# Function to check if last line is unsteady
def CheckUnsteadyHistory(fname='history.dat'):
    """Check if the current history ends with an unsteady iteration

    :Call:
        >>> q = pyCart.case.CheckUnsteadyHistory(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *q*: :class:`float`
            Whether or not the last iteration of *fname* has a '.' in it
    :Versions:
        * 2014-12-17 ``@ddalle``: First version
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return False
    # Check the file's contents.
    try:
        # Try to tail the last line.
        txt = tail(fname)
        # Check for a dot.
        return ('.' in txt.split()[0])
    except Exception:
        # Something failed; invalid history
        return False

    
# Function to get the most recent working folder
def GetWorkingFolder():
    """Get the most recent working folder, either '.' or 'adapt??/'
    
    This function must be called from the top level of a case run directory.
    
    :Call:
        >>> fdir = pyCart.case.GetWorkingFolder()
    :Outputs:
        *fdir*: :class:`str`
            Name of the most recently used working folder with a history file
    :Versions:
        * 2014-11-24 ``@ddalle``: First version
    """
    # Try to get iteration number from working folder.
    n0 = GetHistoryIter()
    # Initialize working directory.
    fdir = '.'
    # Check for adapt?? folders
    for fi in glob.glob('adapt??'):
        # Attempt to read it.
        ni = GetHistoryIter(os.path.join(fi, 'history.dat'))
        # Check it.
        if ni > n0:
            # Current best estimate of working directory.
            fdir = fi
    # Output
    return fdir
       
# Function to get most recent adaptive iteration
def GetCurrentResid():
    """Get the most recent iteration including unsaved progress

    Iteration numbers from time-accurate restarts are corrected to match the
    global iteration numbering.

    :Call:
        >>> L1 = pyCart.case.GetCurrentResid()
    :Outputs:
        *L1*: :class:`float`
            Last L1 residual
    :Versions:
        * 2015-01-02 ``@ddalle``: First version
    """
    # Get the working folder.
    fdir = GetWorkingFolder()
    # Get the residual.
    return GetHistoryResid(os.path.join(fdir, 'history.dat'))
    
# Function to get most recent L1 residual
def GetCurrentIter():
    """Get the residual of the most recent iteration including unsaved progress

    :Call:
        >>> n = pyCart.case.GetCurrentIter()
    :Outputs:
        *n*: :class:`int`
            Most recent index written to :file:`history.dat`
    :Versions:
        * 2014-11-28 ``@ddalle``: First version
    """
    # Try to get iteration number from working folder.
    ntd = GetHistoryIter()
    # Check it.
    if ntd and (not CheckUnsteadyHistory()):
        # Don't read adapt??/ history
        return ntd
    # Initialize adaptive iteration number
    n0 = 0
    # Check for adapt?? folders
    for fi in glob.glob('adapt??'):
        # Attempt to read it.
        ni = GetHistoryIter(os.path.join(fi, 'history.dat'))
        # Check it.
        if ni > n0:
            # Update best estimate.
            n0 = ni
    # Output the total.
    return n0 + ntd
    
    
