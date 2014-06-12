#!/usr/bin/python
"""
Run case setup script: :mod:`pc_SetupRuns`
==========================================

Setup runs based on existing pyCart settings using defaults where necessary.
This script assumes that 'pc_SetupGrids.py' has already been run, or a similar
action has been taken to create mesh files in each grid folder.

:Call:
    .. code-block:: console
    
        $ pc_SetupRuns.py
        $ pc_SetupRuns.py $json
        $ pc_SetupRuns.py -h
    
:Inputs:
    *json*: Name of pyCart control file (defaults to "pyCart.json")
    
:Options:
    *h*: Show this help message and exit
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr

# Primary function
def SetupRuns(*a, **kw):
    """
    Main function for ``pc_SetupRuns.py`` script
    
    :Call:
        >>> SetupRuns(fname='pyCart.json', h=False)
        
    :Inputs:
        *fname*: :class:`str`
            Name of global pyCart settings file
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
    
    # Create the folders.
    cart3d.CreateFolders()
    
    # Copy/link the files.
    cart3d.CopyFiles()
    
    # Process the inputs and create the scripts.
    cart3d.PrepareRuns()


# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function
    SetupRuns(*a, **kw)
    




