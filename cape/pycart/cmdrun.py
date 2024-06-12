r"""
:mod:`cape.pycart.cmdrun`: Cart3D executable interface module
==============================================================

This module provides an interface to the various executables of the
Cart3D package. The functions in this module have names that match the
command-line names of those Cart3D executables.

    * :func:`cubes`: Calls ``cubes``
    * :func:`mgPrep`: Calls ``mgPrep``
    * :func:`autoInputs`: Calls ``autoInputs``
    * :func:`flowCart`: Calls ``flowCart`` or ``mpix_flowCart``

:See also:
    * :mod:`cape.cfdx.cmdgen`
    * :mod:`cape.cfdx.cmdrun`
    * :mod:`cape.pycart.cmdgen`

"""

# Standard library
import os

# Import relevant tools
from .options import Options
from ..cfdx.cmdgen import isolate_subsection
from ..cfdx.cmdrun import *
from ..cfdx.cmdrun import (
    callf,
    _assertfile,
    _upgradeDocString)

# Command option processing
from . import cmdgen


# Function to call cubes.
def cubes(opts=None, j=0, **kwargs):
    # Required file
    _assertfile('input.c3d')
    # Get command
    cmdi = cmdgen.cubes(opts=opts, j=j, **kwargs)
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Get verbose option
    v = opts.get_Verbose(j)
    # Run the command.
    return callf(cmdi, f='cubes.out', v=v)


# Docstring
cubes.__doc__ = _upgradeDocString(cmdgen.cubes.__doc__)


# Function to call mgPrep
def mgPrep(opts=None, j=0, **kwargs):
    # Required file
    _assertfile('Mesh.R.c3d')
    # Get the command.
    cmdi = cmdgen.mgPrep(opts=opts, j=j, **kwargs)
    # Get verbose option
    if opts:
        v = opts.get_Verbose(j)
    else:
        v = True
    # Run the command
    return callf(cmdi, f='mgPrep.out', v=v)


# Docstring
mgPrep.__doc__ = _upgradeDocString(cmdgen.mgPrep.__doc__)


# Function to call mgPrep
def autoInputs(opts=None, j=0, **kwargs):
    # Get command.
    cmdi = cmdgen.autoInputs(opts=opts, j=j, **kwargs)
    # Get verbose option
    if opts:
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
autoInputs.__doc__ = _upgradeDocString(cmdgen.autoInputs.__doc__)


# Function to call flowCart
def flowCart(opts=None, j=0, **kwargs):
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Apply options
    opts.set_opts(**kwargs)
    # Get values from internal settings.
    nProc = opts.get_OMP_NUM_THREADS(j)
    # Set environment variable.
    if nProc:
        os.environ['OMP_NUM_THREADS'] = str(nProc)
    # Get command.
    cmdi = cmdgen.flowCart(opts, j=j)
    # Get verbose option
    v = opts.get_Verbose(j)
    # Run the command
    callf(cmdi, f='flowCart.out', v=v)


# Docstring
flowCart.__doc__ = _upgradeDocString(cmdgen.flowCart.__doc__)

