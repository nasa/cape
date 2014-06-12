#!/usr/bin/python
"""
Post-process "loadsTRI.dat": :mod:`pc_GatherLoadsTRI`
=====================================================

Read all "loadsTRI.dat" files in a trajectory and write the combined results to
"loadsTRI.csv" in the root directory.  Any missing cases or components will
result in ``nan`` values in the combined results file.

:Call:

    .. code-block:: console
    
        $ pc_GatherLoadsTRI.py
        $ pc_GatherLoadsTRI.py $json
        $ pc_GatherLoadsTRI.py -h
    
:Inputs:
    *json*: Name of pyCart control file (defaults to "pyCart.json")

:Options:
    *h*: Display this help and exit
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr

# Main function
def GatherLoadsTRI(*a, **kw):
    """
    Read all :file:`loadsTRI.dat` files from run cases and write combined results
    to :file:`loadsTRI.csv` in the root directory.
    
    :Call:
        
        >>> GatherLoads(fname, h=False)
        
    :Inputs:
        *fname*: :class:`str`
            Name of global pyCart settings file, default is ``'pyCart.json'``
        *h*: :class:`bool`
            If ``True``, show help and exit
    """
    # Versions:
    #  2014.06.12 @ddalle  : First documented version
    
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Default file name.
        fname = 'pyCart.json'
    else:
        # Use the first general input.
        fname = a[0]
        
    # Read in the settings file.
    cart3d = pyCart.Cart3d(fname)
    
    # Read the "loadsTRI.dat" files.
    cart3d.GetLoadsTRI()
    
    # Write the files.
    cart3d.WriteLoadsTRI()

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function
    GatherLoadsTRI(*a, **kw)

