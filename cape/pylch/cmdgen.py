r"""
:mod:`cape.pylava.cmdgen`: Create commands for LAVA executables
====================================================================

This module provides the function :func:`superlava` to generate the
command-line call to use the ``superlava`` executable. This includes
the various aspects of the ``mpiexec`` or similar call that might be
utilized to run ``superlava`` in parallel.
"""

# Standard library
import os
from typing import Optional

# Local imports
from .options import Options
from ..optdict import OptionsDict
from ..cfdx.archivist import getmtime, reglob
from ..cfdx.cmdgen import (
    append_cmd_if,
    isolate_subsection,
    mpiexec)


# Function to create superlava command
def chem(opts: Optional[OptionsDict] = None, j: int = 0, **kw):
    r"""Interface to Loci/CHEM binary

    :Call:
        >>> cmdi = chem(opts, i=0)
    :Inputs:
        *opts*: :class:`Options`
            Options instance, either global or *RunControl*
        *j*: {``0``} | :class:`int`
            Phase number
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2024-10-17 ``@ddalle``: v1.0
    """
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    chemopts = isolate_subsection(opts, Options, ("RunControl", "chem"))
    # Initialize with MPI portion of command
    cmdi = mpiexec(opts, j)
    # Get name of executable
    execname = chemopts.get_opt("executable", j=j)
    # Get name of project
    project = opts.get_opt("ProjectName", j=j)
    # Append to command "chem pylch" or similar
    cmdi.append(execname)
    cmdi.append(project)
    # Check for restart
    r = find_restart_iter()
    # Append if applicable
    append_cmd_if(cmdi, r, [r])
    # Output
    return cmdi


# Find restarts
def find_restart_iter() -> Optional[str]:
    r"""Find latest Loci/CHEM restart number within current folder

    :Call:
        >>> r = find_restart_iter()
    :Outputs:
        *r*: :class:`str`
            Text of an integer of the most recent restart folder
    :Versions:
        * 2024-10-17 ``@ddalle``: v1.0
    """
    # Find canidate folders
    restart_list = reglob("restart/[0-9]+")
    # Check for null restart
    if len(restart_list) == 0:
        return
    # Sort by modification time
    restart_sorted = sorted(restart_list, key=getmtime)
    # Return the last one, if any
    return os.path.basename(restart_sorted[-1])
