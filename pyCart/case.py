"""
Case for interacting with individual run cases: :mod:`pyCart.case`
==================================================================

"""

# Import options class
from options.flowCart import flowCart
# Binary interface.
from bin import callf
# Interface for writing commands
import cmd

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
    # Get the settings.
    fc = ReadJSON()
    # Determine the run index.
    i = DetermineInputNumber(fc)
    # Get the restart iteration number.
    n = GetRestartIter()
    # Get the number of threads, which may be irrelevant.
    nThread = fc.get_OMP_NUM_THREADS()
    # Set it.
    os.environ['OMP_NUM_THREADS'] = str(nThread)
    # Make the command.
    cmdi = cmd.flowCart(fc=fc, i=i, n=n)
    # Run the command.
    callf(cmdi, f='flowCart.out')
    # Assuming that worked, move the temp output file.
    os.rename('flowCart.out', 'run.%02i.%i' % (i, n+fc.get_it_fc(i)))
    

# Function to read the local settings file.
def ReadJSON():
    """Read `flowCart` settings for local case
    
    :Call:
        >>> fc = pyCart.case.ReadJSON()
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
    fch = glob.glob('check.*')
    # Initialize iteration number until informed otherwise.
    n = 0
    # Check its contents.
    if len(fch) == 0:
        # Empty, no restart available.
        return n
    # Loop through the matches.
    for fname in fch:
        # Get the integer for this file.
        i = int(fname[6:])
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
    # Case completed; just return the last value.
    return i
    
    
