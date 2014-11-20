"""
Case for interacting with individual run cases: :mod:`pyCart.case`
==================================================================

"""

# Import options class
from options.flowCart import flowCart
# Binary interface.
from bin import callf
# Interface for writing commands
from . import cmd, queue

# Read the local JSON file.
import json
# File control
import os, glob


# Function to setup and call the appropriate flowCart file.
def run_flowCart():
    """
    Setup and run the appropriate `flowCart`, `mpi_flowCart` command
    
    :Call:
        >>> pyCart.case.run_flowCart()
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
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
    i = DetermineInputNumber(fc)
    # Get the restart iteration number.
    n = GetRestartIter()
    # Delete any input file.
    if os.path.isfile('input.cntl'): os.remove('input.cntl')
    # Create the correct input file.
    os.symlink('input.%02i.cntl' % i, 'input.cntl')
    # Extra prep for adaptive --> non-adaptive
    if (i>0) and (not fc.get_use_aero_csh(i)) and (os.path.isdir('BEST')
            and (not os.path.isfile('history.dat'))):
        # Go to the best adaptive result.
        os.chdir('BEST')
        # Find all *.dat files and Mesh files
        fglob = glob.glob('*.dat') + glob.glob('Mesh.*.c3d')
        # Go back up one folder.
        os.chdir('..')
        # Copy all the important files.
        for fname in fglob:
            # Check for the file.
            if os.path.isfile(fname):
                # Move it to some other file.
                os.path.rename(fname, fname+".old")
            # Copy the file.
            shutil.copy('BEST/'+fname, fname)
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
        if n > 0:
            # Restart case.
            cmdi = ['./aero.csh', 'restart']
        else:
            # Initial case
            cmdi = ['./aero.csh', 'jumpstart']
    else:
        # Call flowCart directly.
        cmdi = cmd.flowCart(fc=fc, i=i, n=n)
    # Run the command.
    callf(cmdi, f='flowCart.out')
    # Get the new restart iteration.
    n = GetRestartIter()
    # Assuming that worked, move the temp output file.
    os.rename('flowCart.out', 'run.%02i.%i' % (i, n))
    # Check for TecPlot files to save.
    if os.path.isfile('cutPlanes.plt'):
        os.rename('cutPlanes.plt', 'cutPlanes.%02i.plt' % i)
    if os.path.isfile('Components.i.plt'):
        os.rename('Components.i.plt', 'Components.%02i.plt' % i)
    # Remove the RUNNING file.
    if os.path.isfile('RUNNING'): os.remove('RUNNING')
    # Check current iteration count.
    if n >= fc.get_LastIter():
        return
    # Resubmit if asked.
    if fc.get_resub(i):
        # Run full restart command, including qsub if appropriate
        StartCase()
    else:
        # Just run the case directly (keep the same PBS job).
        callf(['bash', 'run_cart3d.pbs'])
    
    
# Function to call script or submit.
def StartCase():
    """Start a case by either submitting it or calling with a system command
    
    :Call:
        >>> pyCart.case.StartCase()
    :Versions:
        * 2014.10.06 ``@ddalle``: First version
    """
    # Get the config.
    fc = ReadCaseJSON()
    # Determine the run index.
    i = DetermineInputNumber(fc)
    # Check qsub status.
    if fc.get_qsub(i):
        # Submit the case.
        pbs = queue.pqsub('run_cart3d.pbs')
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_flowCart()
    

# Function to read the local settings file.
def ReadCaseJSON():
    """Read `flowCart` settings for local case
    
    :Call:
        >>> fc = pyCart.case.ReadCaseJSON()
    :Outputs:
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Options interface for `flowCart`
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
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
def GetRestartIter():
    """Get iteration number of most recent check file
    
    :Call:
        >>> n = pyCart.case.GetRestartIter()
    :Outputs:
        *n*: :class:`int`
            Index of most recent check file
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
    """
    # List the check.* files.
    fch = glob.glob('check.*') + glob.glob('BEST/check.*')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Check its contents.
    if len(fch) == 0:
        # Empty, no restart available.
        return n
    # Loop through the matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname.split('.')[-1])
        # Use the running maximum.
        n = max(i, n)
    # Output.
    return n
    
# Function to set up most recent check file as restart.
def SetRestartIter(n=None):
    """Set a given check file as the restart point
    
    :Call:
        >>> pyCart.case.SetRestartIter(n=None)
    :Inputs:
        *n*: :class:`int`
            Restart iteration number, defaults to most recent available
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
    """
    # Check the input.
    if n is None:
        n = GetRestartIter()
    # Remove the current restart file if necessary.
    if os.path.isfile('Restart.file'):
        os.remove('Restart.file')
    # Quit if no check point.
    if n == 0:
        return None
    # Create a link to the file.
    if os.path.isfile('check.%05i' % n):
        os.symlink('check.%05i' % n, 'Restart.file')
    
    
# Function to chose the correct input to use from the sequence.
def DetermineInputNumber(fc):
    """Determine the appropriate input number based on results available
    
    :Call:
        >>> i = pyCart.case.DetermineInputNumber(fc)
    :Inputs:
        *fc*: :class:`pyCart.options.flowCart.flowCart`
            Options interface for `flowCart`
    :Outputs:
        *i*: :class:`int`
            Most appropriate run number for a restart
    :Versions:
        * 2014.10.02 ``@ddalle``: First version
    """
    # Get the run index.
    n = GetRestartIter()
    # Set the restart file if appropriate.
    SetRestartIter(n)
    # Loop through possible input numbers.
    for i in range(fc.get_nSeq()):
        # Check for output files.
        if not glob.glob('run.%02i.*' % i):
            # This run has not been completed yet.
            return i
        # Check the iteration number.
        if n < fc.get_IterSeq(i):
            # This case has been run, but hasn't reached the min iter cutoff
            return i
    # Set the restart file if appropriate.
    if not fc.get_use_aero_csh(i):
        SetRestartIter(n)
    # Case completed; just return the last value.
    return i
    
    
