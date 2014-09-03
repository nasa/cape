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
        ierr = sp.call(cmd, stdout=fid, shell=shell)
        # Close the file.
        fid.close()
    else:
        # Call the command.
        ierr = sp.call(cmd, shell=shell)
    # Check the status.
    if ierr:
        raise SystemError("Command failed with status %i." % ierr)

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
def cubes(cart3d=None, maxR=10, reorder=True, pre='preSpec.c3d.cntl',
    cubes_a=None, cubes_b=None):
    """
    Interface to Cart3D script `cubes`
    
    :Call:
        >>> pyCart.bin.cubes(cart3d)
        >>> pyCart.bin.cubes(maxR=10, reorder=True, pre='preSpec.c3d.cntl')
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *maxR*: :class:`int`
            Number of refinements to make
        *reorder*: :class:`bool`
            Whether or not to reorder mesh
        *pre*: :class:`str`
            Name of prespecified bounding box file (or ``None``)
        *cubes_a*: :class:`int`
            `cubes` "a" parameter
        *cubes_b*: :class:`int`
            `cubes` "b" parameter
    :Versions:
        * 2014.06.30 ``@ddalle``: First version
        * 2014.09.02 ``@ddalle``: Rewritten with new options paradigm
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        maxR    = cart3d.opts.get_maxR()
        pre     = cart3d.opts.get_pre()
        reorder = cart3d.opts.get_reorder()
        cubes_a = cart3d.opts.get_cubes_a()
        cubes_b = cart3d.opts.get_cubes_b()
    # Required file
    _assertfile('input.c3d')
    # Initialize command
    cmd = ['cubes']
    # Add options.
    if maxR:    cmd += ['-maxR', str(maxR)]
    if reorder: cmd += ['-reorder']
    if pre:     cmd += ['-pre', pre]
    if cubes_a: cmd += ['-cubes_a', str(cubes_a)]
    if cubes_b: cmd += ['-cubes_b', str(cubes_b)]
    # Run the command.
    callf(cmd, f='cubes.out')
    
# Function to call mgPrep
def mgPrep(cart3d=None, mg_fc=3):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> pyCart.bin.mgPrep(cart3d)
        >>> pyCart.bin.mgPrep(mg_fc=3)
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *mg_fc*: :class:`int`
            Number of multigrid levels to prepare
    :Versions:
        * 2014.09.02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        mg_fc = cart3d.opts.get_mg_fc()
    # Required file
    _assertfile('Mesh.R.c3d')
    # Initialize command.
    cmd = ['mgPrep']
    # Add options.
    if mg_fc: cmd += ['-n', str(mg_fc)]
    # Run the command.
    callf(cmd, f='mgPrep.out')
    
# Function to call mgPrep
def autoInputs(cart3d=None, r=8, ftri='Components.i.tri'):
    """
    Interface to Cart3D script `mgPrep`
    
    :Call:
        >>> pyCart.bin.autoInputs(cart3d)
        >>> pyCart.bin.autoInputs(r=8, ftri='components.i.tri')
    :Inputs:
        *cart3d*: :class:`pyCart.cart3d.Cart3d`
            Global pyCart settings instance
        *r*: :class:`int`
            Mesh radius
        *ftri*: :class:`str`
            Name of surface triangulation file
    :Versions:
        * 2014.09.02 ``@ddalle``: First version
    """
    # Check cart3d input.
    if cart3d is not None:
        # Apply values
        mg_fc = cart3d.opts.get_r()
    # Initialize command.
    cmd = ['autoInputs']
    # Add options.
    if r:   cmd += ['-n', str(mg_fc)]
    if tri: cmd += ['-t', ftri)
    # Run the command.
    callf(cmd, f='autoInputs.out')
    
    
    
# Function to call flowCart
    

