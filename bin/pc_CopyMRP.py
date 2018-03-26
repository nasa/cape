#!/usr/bin/env python
"""
Copy one component's MRP to another: :file:`pc_CopyMRP.py`
==========================================================

Go through folders and copy the MRP from one component to another component.

:Usage:
    .. code-block:: bash
    
        $ pc_CopyMRP.py $COMP1 $COMP2 [options]
        
:Inputs:
    
    *COMP1*: Name of component to copy MRP from
    *COMP2*: Name of component whose MRP will be changed
    
:Options:

    -h, --help
        Display this help message and quit
        
    -f FNAME
        Use pyCart input file *FNAME* (defaults to 'pyCart.json')
        
    --cons CONS
        Only consider cases that pass a list of inequalities separated by
        commas.  Constraints must use variable names (not abbreviations) from
        the trajectory described in *FNAME*.

:Versions:
    * 2015-03-23 ``@ddalle``: First version
"""

# Import the full module
import pyCart
# Input parsing
from cape.argread import readkeys
# File control
import os, glob
# Numerics
from numpy import sqrt

# Function to fix an individual case
def CopyCaseMRP(cart3d, comp1, comp2):
    """
    Update ``dat`` file(s) for a given component in the current folder if the
    most appropriate :file:`input.cntl` file has a MRP for that component that
    does not match that *cart3d* configuration.
    
    Only the files used for the pyCart data book are updated.
    
    :Call:
        >>> CopyCaseMRP(cart3d, comp1, comp2)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Master Cart3D interface instance
        *comp1*: :class:`str`
            Name of component that will be copied from
        *comp2*: :class:`str`
            Name of component that will be copied to
    :Versions:
        * 2015-03-23 ``@ddalle``: First version
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
    xi = IC.GetSingleMomentPoint(comp2)
    # Get what the MRP should be.
    xo = IC.GetSingleMomentPoint(comp1)
    # Get the distance between the two.
    L = sqrt((xo[0]-xi[0])**2 + (xo[1]-xi[1])**2 + (xo[2]-xi[2])**2)
    # Reference length
    Lref = cart3d.opts.get_RefLength()
    # Check the distance.
    if L/Lref <= 0.01: return
    # Process the best data folder.
    fdir = pyCart.case.GetWorkingFolder()
    # Check for the file
    if not os.path.isfile(os.path.join(fdir, '%s.dat'%comp2)): return
    # Write.
    print("  Updating MRP '%s': %s -> %s" % (comp2, xi, xo))
    # Read the force and moment history for that component.
    FM = pyCart.Aero([comp2])[comp2]
    # Shift the MRP.
    FM.ShiftMRP(Lref, xo, xi)
    # Overwrite the original data file.
    FM.Write(os.path.join(fdir, '%s.dat'%comp2))
    
    # Set the correct value in the file.
    IC.SetSingleMomentPoint(xo, comp2)
    # Write the corrected input file.
    IC.Write(fcntl)
    
    
# Check if run as a script.
if __name__ == "__main__":
    # Parse inputs.
    a, kw = readkeys(pyCart.os.sys.argv)
    
    # Check for a help flag.
    if kw.get('h') or kw.get('help'):
        import cape.text
        print(cape.text.markdown(__doc__))
        pyCart.os.sys.exit()
        
    # Check for adequate components.
    if len(a) != 2:
        print(__doc__)
        raise IOError("Need exactly two components as inputs.")
        
    # Unpack the components
    comp1, comp2 = a
        
    # Get file name.
    fname = kw.get('f', 'pyCart.json')
    
    # Try to read it.
    cart3d = pyCart.Cart3d(fname)
    
    # Get constraints and convert text to list.
    cons  = kw.get('cons',        '').split(',')
    cons += kw.get('constraints', '').split(',')
    # Set the constraints back into the keywords.
    kw['cons'] = cons
    # Check for index list
    if 'I' in kw:
        kw['I'] = cart3d.x.ExpandIndices(kw['I'])
    
    # Apply the constraints.
    I = cart3d.x.GetIndices(**kw)
    # Get the case names.
    fruns = cart3d.x.GetFullFolderNames(I)
    
    # Loop through the runs.
    for frun in fruns:
        # Go to root directory.
        os.chdir(cart3d.RootDir)
        # Check for folder.
        if not os.path.isdir(frun): continue
        # Go to the folder.
        os.chdir(frun)
        # Status update
        print(frun)
        # Update the data.
        CopyCaseMRP(cart3d, comp1, comp2)
    
