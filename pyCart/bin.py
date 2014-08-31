"""
Cart3D binary interface module: :mod:`pyCart.bin`
=================================================

"""

# File system and operating system management
import os
import subprocess as sp

# Function to call commands with a different STDOUT
def callf(cmd, f=None, shell=None):
    """Call a command with alternate STDOUT by filename
    
    :Call:
        >>> callf(cmd, f=None, shell=None)
    :Inputs:
        *cmd*: :class:`list` (:class:`str`)
            List of strings as for :func:`subprocess.call`
        *f*: :class:`str`
            File name to which to store STDOUT
        *shell*: :class:`bool`
            Whether or not a shell is needed
    :Versions:
        * 2014.08.30 ``@ddalle``: First version
    """
    # Process the shell option
    shell = bool(shell)
    # Print what's up
    print(" > " + " ".join(cmd))
    # Check for an output
    if f:
        # Open the file.
        fid = open(f, 'a')
        # Call the command.
        sp.call(cmd, stdout=fid, shell=shell)
        # Close the file.
        fid.close()
    else:
        # Call the command.
        sp.call(cmd, shell=shell)

# Simple function to make sure a file is present
def _assertfile(fname):
    """
    Assert that a given file exists and raise an exception otherwise
    
    :Call:
        >>> _assertfile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to test
    :Versions:
        * 2014.06.30 ``@ddalle``: First version
    """
    # Check for the file.
    if not os.path.isfile(fname):
        raise IOError("No input file '%s' found." % fname)

# Function to call cubes.
def cubes(cart3d=None, maxR=10, reorder=True, pre='preSpec.c3d.cntl'):
    """
    Interface to Cart3D script `cubes`
    
    :Call:
        >>> pyCart.cubes(cart3d)
        >>> pyCart.cubes(maxR=10, reorder=True, pre='preSpec.c3d.cntl')
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *maxR*: :class:`int`
            Number of refinements to make
        *reorder*: :class:`bool`
            Whether or not to reorder mesh
        *pre*: :class:`str`
            Name of prespecified bounding box file (or ``None``)
    :Versions:
        * 2014.06.30 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        maxR    = cart3d.opts.get_maxR()
        pre     = cart3d.opts.get_pre()
        reorder = cart3d.opts.get_reorder()
    # Required file
    _assertfile('input.c3d')
    # Initialize command
    cmd = ['cubes']
    # Add options.
    if maxR:    cmd += ['-maxR', maxR]
    if reorder: cmd += ['-reorder']
    if pre:     cmd += ['-pre', pre]
    # Run the command.
    callf(cmd)
    

