r"""
:mod:`cape.pylava.cmdgen`: Create commands for LAVA executables
====================================================================

This module provides the function :func:`superlava` to generate the
command-line call to use the ``superlava`` executable. This includes
the various aspects of the ``mpiexec`` or similar call that might be
utilized to run ``superlava`` in parallel.
"""

# Standard library
from typing import Optional

# Local imports
from .options import Options
from ..optdict import OptionsDict
from ..cfdx.cmdgen import (
    isolate_subsection,
    infix_phase,
    mpiexec)


# Function to create superlava command
def superlava(opts: Optional[OptionsDict] = None, j: int = 0, **kw):
    r"""Interface to LAVACURV binary

    :Call:
        >>> cmdi = superlava(opts, i=0)
        >>> cmdi = superlava(**kw)
    :Inputs:
        *opts*: :class:`Options`
            Options instance, either global or *RunControl*
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2024-10-10 ``@ddalle``: v1.0
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    lavaopts = isolate_subsection(opts, Options, ("RunControl", "superlava"))
    # Initialize with MPI portion of command
    cmdi = mpiexec(opts, j)
    # Get name of executable
    execname = lavaopts.get_opt("executable")
    # Get name of YAML file
    yamlfile = lavaopts.get_opt("yamlfile")
    # Append to command
    cmdi.append(execname)
    cmdi.append(infix_phase(yamlfile, j))
    # Output
    return cmdi

