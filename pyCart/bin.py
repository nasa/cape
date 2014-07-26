"""
Cart3D binary interface module: :mod:`pyCart.bin`
=================================================

"""

# File system and operating system management
import os

# Simple function to make sure a file is present
def _assertfile(fname):
    """
    Assert that a given file exists and raise an exception otherwise
    
    :Call:
        >>> _assertfile(fname)
        
    :Inputs:
        *fname*: :class:`str`
            Name of file to test
    """
    # Versions:
    #  2014.06.30 @ddalle  : First version
    
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
        *cubes
    """
    # Versions:
    #  2014.06.30 @ddalle  : First version
    
    # Check cart3d input.
    if cart3d is not None:
        # Specified parameters
        Mesh = cart3d.Options['Mesh']
        # Get the values.
        maxR = Mesh.get('nRefinements', maxR)
        pre  = 'preSpec.c3d.cntl'
        reorder = True
    # Required file
    _assertfile('input.c3d')
    # Initialize command
    cmd = 'cubes -maxR %i' % maxR
    # Add the reorder flag if appropriate.
    if reorder:
        cmd += ' -reorder'
    # Add the preSpec option if appropriate.
    if pre and os.path.isfile(pre):
        cmd += (' -pre %s' % pre)
    # Run the command.
    os.system(cmd)
