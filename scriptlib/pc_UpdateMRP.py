#!/usr/bin/env python
"""
Fix cases' moment reference points: :file:`pc_UpdateMRP.py`
===========================================================

Go through folders and ensure that forces and moments match the settings in the
pyCart master JSON file.

:Usage:
    .. code-block:: bash
    
        $ pc_UpdateMRP.py $COMP [options]
        
:Inputs:
    
    *COMP*: Name of component to check and/or fix
    
:Options:

    -h, --help
        Display this help message and quit
        
    -f FNAME
        Use pyCart input file *FNAME* (defaults to 'pyCart.json')
        
    -p [COMP]
        Create multi-page PDF plots according to settings in *FNAME*.  If 
        *COMP* is specified, only that component is plotted
        
    --cons CONS
        Only consider cases that pass a list of inequalities separated by
        commas.  Constraints must use variable names (not abbreviations) from
        the trajectory described in *FNAME*.

:Versions:
    * 2015-03-02 ``@ddalle``: First version
"""

# Import the full module
import pyCart
# Input parsing
from pyCart.argread import readkeys
# File control
import os, glob
# Numerics
from numpy import sqrt

# Function to fix an individual case
def UpdateCaseMRP(cart3d, comp):
    """
    Update ``dat`` file(s) for a given component in the current folder if the
    most appropriate :file:`input.cntl` file has a MRP for that component that
    does not match that *cart3d* configuration.
    
    Only the files used for the pyCart data book are updated.
    
    :Call:
        >>> UpdateCaseMRP(cart3d, comp, x)
    :Inputs:
        *cart3d*: 
    :Versions:
        * 2015-03-02 ``@ddalle``: First version
    """
    # List sequential input.cntl files
    fcntl = glob.glob('input.??.cntl')
    # Check for matches
    if len(fcntl) > 0:
        # Get latest.
        ncntl = max([int(f.split('.')[1]) for f in fcntl])
        # Form the file.
        fcntl = 'input.%02i.cntl' % ncntl
    else:
        # Use the base file
        fcntl = 'input.cntl'
    # Check for the file.
    if not os.path.isfile(fcntl): return
    # Check for failure or RUNNING
    if os.path.isfile('RUNNING') or os.path.isfile('FAIL'): return
    # Open the input control file interface.
    IC = pyCart.InputCntl(fcntl)
    # Get the MRP for that component that was actually used
    xi = IC.GetSingleMomentPoint(comp)
    # Get what the MRP should be.
    x = cart3d.opts.get_RefPoint(comp)
    # Get the distance between the two.
    L = sqrt((x[0]-xi[0])**2 + (x[1]-xi[1])**2 + (x[2]-xi[2])**2)
    # Reference length
    Lref = cart3d.opts.get_RefLength()
    # Check the distance.
    if L/Lref <= 0.01: return
    # Write.
    print("  Updating MRP '%s': %s -> %s" % (comp, xi, x))
    # Read the force and moment history for that component.
    FM = pyCart.Aero([comp])[comp]
    # Shift the MRP.
    FM.ShiftMRP(Lref, x, xi)
    # Process the best data folder.
    fdir = pyCart.case.GetWorkingFolder()
    # Overwrite the original data file.
    FM.Write(os.path.join(fdir, '%s.dat'%comp))
    
    # Set the correct value.
    IC.SetSingleMomentPoint(x, comp)
    # Write the corrected input file.
    IC.Write(fcntl)
    
    
# Check if run as a script.
if __name__ == "__main__":
    # Parse inputs.
    a, kw = readkeys(pyCart.os.sys.argv)
    
    # Get constraints and convert text to list.
    cons  = kw.get('cons',        '').split(',')
    cons += kw.get('constraints', '').split(',')
    # Set the constraints back into the keywords.
    kw['cons'] = cons
    
    # Check for a help flag.
    if kw.get('h') or kw.get('help'):
        print(__doc__)
        pyCart.os.sys.exit()
        
    # Get file name.
    fname = kw.get('f', 'pyCart.json')
    
    # Try to read it.
    cart3d = pyCart.Cart3d(fname)
    
    # Apply the constraints.
    I = cart3d.x.Filter(cons)
    # Get the case names.
    fruns = cart3d.x.GetFullFolderNames(I)
    
    # Loop through the runs.
    for frun in fruns:
        # Change to the directory.
        os.chdir(cart3d.RootDir)
        os.chdir(frun)
        # Status update
        print(frun)
        # Loop through components.
        for comp in a:
            # Update the data.
            UpdateCaseMRP(cart3d, comp)
    
