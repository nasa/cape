"""
:mod:`pyCart.bin`: Cart3D binary interface module
=================================================

This module provides an interface to the various executables of the Cart3D
package.  The functions in this module have names that match the command-line
names of those Cart3D executables.

    * :func:`cubes`: Calls ``cubes``
    * :func:`mgPrep`: Calls ``mgPrep``
    * :func:`autoInputs`: Calls ``autoInputs``
    * :func:`flowCart`: Calls ``flowCart`` or ``mpix_flowCart``
    
In addition, all the more universal executable interfaces provided in
:mod:`cape.bin`, including the Cart3D utilities ``intersect`` and ``verify``,
are also imported.  These are imported directly

    .. code-block:: python
    
        from cape.bin import *
        
so no extra syntax is needed in order to access them from :mod:`pyCart.bin`

:See also:
    * :mod:`cape.bin`
    * :mod:`cape.cmd`
    * :mod:`pyCart.cmd`

"""

# Import relevant tools
from cape.bin import *
from cape.bin import _assertfile, _upgradeDocString

# Command option processing
import cmd

# Function to call cubes.
def cubes(cart3d=None, opts=None, j=0, **kwargs):
    # Required file
    _assertfile('input.c3d')
    # Get command
    cmdi = cmd.cubes(cart3d=cart3d, opts=opts, j=j, **kwargs)
    # Get verbose option
    if cart3d:
        v = cart3d.opts.get_Verbose(j)
    elif opts:
        v = opts.get_Verbose(j)
    else:
        v = True
    # Run the command.
    callf(cmdi, f='cubes.out', v=v)
# Docstring
cubes.__doc__ = _upgradeDocString(cmd.cubes.__doc__)
    
# Function to call mgPrep
def mgPrep(cart3d=None, opts=None, j=0, **kwargs):
    # Required file
    _assertfile('Mesh.R.c3d')
    # Get the command.
    cmdi = cmd.mgPrep(cart3d=cart3d, opts=opts, j=j, **kwargs)
    # Get verbose option
    if cart3d:
        v = cart3d.opts.get_Verbose(j)
    elif opts:
        v = opts.get_Verbose(j)
    else:
        v = True
    # Run the command.
    callf(cmdi, f='mgPrep.out', v=v)
# Docstring
mgPrep.__doc__ = _upgradeDocString(cmd.mgPrep.__doc__)
    
# Function to call mgPrep
def autoInputs(cart3d=None, opts=None, j=0, **kwargs):
    # Get command.
    cmdi = cmd.autoInputs(cart3d, opts=opts, j=j, **kwargs)
    # Get verbose option
    if cart3d:
        v = cart3d.opts.get_Verbose(j)
    elif opts:
        v = opts.get_Verbose(j)
    else:
        v = True
    # Run the command.
    callf(cmdi, f='autoInputs.out', v=v)
    # Fix the name of the triangulation in the 'input.c3d' file
    # Read the intersect file.
    lines = open('input.c3d').readlines()
    # Change the triangulation file
    lines[7] = '  Components.i.tri\n'
    # Write the corrected file.
    open('input.c3d', 'w').writelines(lines)
# Docstring
autoInputs.__doc__ = _upgradeDocString(cmd.autoInputs.__doc__)
    
    
    
# Function to call flowCart
def flowCart(cart3d=None, i=0, **kwargs):
    # Check for cart3d input
    if cart3d is not None:
        # Get values from internal settings.
        nProc   = cart3d.opts.get_OMP_NUM_THREADS(i)
    else:
        # Get values from keyword arguments
        nProc   = kwargs.get('nProc', 4)
    # Set environment variable.
    if nProc:
        os.environ['OMP_NUM_THREADS'] = str(nProc)
    # Get command.
    cmdi = cmd.flowCart(cart3d=cart3d, i=i, **kwargs)
    # Get verbose option
    if cart3d:
        v = cart3d.opts.get_Verbose(j)
    elif opts:
        v = opts.get_Verbose(j)
    else:
        v = True
    # Run the command
    callf(cmdi, f='flowCart.out', v=v)
# Docstring
flowCart.__doc__ = _upgradeDocString(cmd.flowCart.__doc__)

