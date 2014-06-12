#!/usr/bin/python
"""
Run case script template: :mod:`pc_CaseTemplate`
================================================

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
import sys, os, shutil
# Command-line input parser
import pyCart.argread as argr

# Numeric functions
import numpy as np

# Primary function
def main(*a, **kw):
    """
    Main function for this script
    
    :Call:
        >>> main(fname='pyCart.json')
        
    :Inputs:
        *fname*: :class:`str`
            Name of global pyCart settings file
    """
    # Versions:
    #  2014.06.12 @ddalle  : First documented version
    
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Default file name.
        fname = kw.get('fname', 'pyCart.json')
    else:
        # Use the first general input.
        fname = a[0]
        
    # Read in the settings file.
    cart3d = pyCart.Cart3d(fname)
    
    # Get the trajectory.
    T = cart3d.Trajectory
    
    # Read the "input.cntl" file
    ic = pyCart.InputCntl()
    
    # # Example global task: read the tri file(s)
    # if type(cart3d.Mesh['TriFile']) is list:
    #     # List of filenames
    #     ftri = [f for f in cart3d.Mesh['TriFile']]
    #     # Read each file.
    #     tri = [cart3d.Tri(f) for f in cart3d.Mesh['TriFile']]
    # else:
    #     # Make a list of one tri file.
    #     ftri = [cart3d.Mesh['TriFile']]
    #     tri = [cart3d.Tri(cart3d.Mesh['TriFile'])]
    
    # Get a list of the unique grid folders.
    glist = np.unique(T.GetGridFolderNames())
    # Get a list of the paths to the case folder from root.
    dlist = T.GetFullFolderNames()
    
    # Loop through the meshes.
    for i in range(len(glist)):
        # Perform folder-specific functions...
        # Example: make the folder
        os.mkdir(glist[i], 0750)
        # Go to the folder.
        os.chdir(glist[i])
        # Do more stuff...
        
        # Example: write the "input.cntl" file
        # ic.Write()
        
        # Example: write the tri files.
        # # Rotate the second tri by 10 degrees about x-axis.
        # tri[1].Rotate([0,0,0], [0,0,1.], 10)
        # # Write all
        # for i in range(len(tri)):
        #     tri[i].Write(ftri[i])
        
        # Go back to the root folder.
        os.chdir('..')
    
    # # Example: use the built-in function to copy files to case dirs.
    # # Copy/link the files.
    # cart3d.CopyFiles()
    
    # Loop through the cases.
    for i in range(len(dlist)):
        # Perform case-specific functions...
        # Example: make the folder
        os.mkdir(dlist[i], 0750)
         # Go to the folder.
        os.chdir(dlist[i])
        # Do more stuff...
        
        # Example: set Mach number, angle of attack, and write new "input.cntl"
        # # Set conditions
        # ic.SetMach(T.Mach[i])
        # ic.SetAlpha(T.Alpha[i])
        # ic.SetBeta(T.Beta[i])
        # # Do more stuff ... like set a boundary condition on component 2
        # ic.SetSurfBC(2, [1., 2.2, 0.03, 0., 7.43])
        # 
        # # Write the file.
        # ic.Write()
        
        # Example: create a specific run script.
        # # Create the file.
        # f = open('run_cart.sh')
        # # Write header.
        # f.write('#!/bin/bash\n\n')
        # # Set thread count
        # f.write('export OMP_NUM_THREADS=8\n\n')
        # # Call flowCart.
        # f.write('flowCart -N %i -v -mg 3 -his -clic' % cart3d.Options['nIter'])
        # # Close the file.
        # f.close()
        
        # Go back to the root folder.
        os.chdir('..')
        os.chdir('..')


# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function
    main(*a, **kw)
    




