"""
Cart3D binary interface module: :mod:`pyCart.bin`
=================================================

Direct system interfaces to Cart3D command-line tools
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
    # Run the command.
    callf(cmdi, f='cubes.out')
# Docstring
cubes.__doc__ = _upgradeDocString(cmd.cubes.__doc__)
    
# Function to call mgPrep
def mgPrep(cart3d=None, opts=None, j=0, **kwargs):
    # Required file
    _assertfile('Mesh.R.c3d')
    # Get the command.
    cmdi = cmd.mgPrep(cart3d=cart3d, opts=opts, j=j, **kwargs)
    # Run the command.
    callf(cmdi, f='mgPrep.out')
# Docstring
mgPrep.__doc__ = _upgradeDocString(cmd.mgPrep.__doc__)
    
# Function to call mgPrep
def autoInputs(cart3d=None, opts=None, j=0, **kwargs):
    # Get command.
    cmdi = cmd.autoInputs(cart3d, opts=opts, j=j, **kwargs)
    # Run the command.
    callf(cmdi, f='autoInputs.out')
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
    # Run the command
    callf(cmdi, f='flowCart.out')
# Docstring
flowCart.__doc__ = _upgradeDocString(cmd.flowCart.__doc__)

# Function to call Tecplot on a macro
def tecmcr(mcr="export-lay.mcr", **kwargs):
    # Get command.
    cmdi = cmd.tecmcr(mcr, **kwargs)
    # Run the command.
    callf(cmdi, f="tecmcr.out")
    # Remove the log file; it's useless
    os.remove("tecmcr.out")
# Docstring
tecmcr.__doc__ = _upgradeDocString(cmd.tecmcr.__doc__)

