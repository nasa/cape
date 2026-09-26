r"""
:mod:`cape.pyvul.cmdgen`: Create commands for VULCAN-CFD executables
====================================================================

This module creates system commands as lists of strings for the
VULCAN-CFD executable ``vulcan``. Unlike most solvers, ``vulcan`` is a
C-shell script that handles the MPI launch internally, so the number of
processors and host file are positional arguments rather than
``mpiexec`` flags:

    .. code-block:: bash

        vulcan <num cpus> <host file> <options> <inp file> [out file]

Commands are created in the form of a list of strings.  This is the
format used in the built-in module :mod:`subprocess` and also with
:func:`cape.bin.calli`. For example,

    .. code-block:: python

        ["vulcan", "24", "NULL", "-s", "vulcan.inp", "vulcan.out"]

:See also:
    * :mod:`cape.cfdx.cmdgen`
    * :mod:`cape.pyvul.options.runctlopts`
"""

# Standard library
import os

# Local imports
from .options import Options, runctlopts
from ..cfdx.cmdgen import get_nproc, isolate_subsection
from ..optdict.optitem import getel

# Single-character VULCAN execution flags, in command-line order
_VULCAN_FLAG_ORDER = (
    ("pre", "p"),
    ("solve", "s"),
    ("post", "g"),
    ("recompose", "r"),
)

# Options handled explicitly in :func:`vulcan`
_VULCAN_SKIP = (
    "run", "nproc", "hostfile", "inpfile", "outfile",
    "pre", "solve", "post", "recompose",
)


# Function to create the ``vulcan`` command
def vulcan(opts=None, j=0, **kw):
    r"""Interface to the VULCAN-CFD script ``vulcan``

    :Call:
        >>> cmdi = vulcan(opts, j=0)
        >>> cmdi = vulcan(**kw)
    :Inputs:
        *opts*: :class:`cape.pyvul.options.Options`
            Global pyVul options interface or "RunControl" interface
        *j*: :class:`int`
            Phase number
        *kw*: :class:`dict`
            Additional options applied over the "vulcan" settings;
            keys not in ``VulcanOpts._optlist`` are appended as
            single-dash CLI options
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2026-09-25 ``@ddalle``: v1.0
    """
    # Isolate opts for "RunControl" section
    if isinstance(opts, Options):
        # Downselect to "RunControl" section
        opts = isolate_subsection(opts, Options, ("RunControl",))
    elif isinstance(opts, dict) and "RunControl" in opts:
        # Raw dictionary input
        opts = opts["RunControl"]
    # Get vulcan options
    if hasattr(opts, "get_MPI"):
        # Full "RunControl" interface
        q_mpi = opts.get_MPI(j)
        vopts = opts["vulcan"]
    else:
        if isinstance(opts, dict) and ("vulcan" not in opts) and (
                "MPI" not in opts) and ("nProc" not in opts):
            # Raw "vulcan" section dictionary
            opts = dict(vulcan=opts)
        # Raw "RunControl" dictionary
        opts = runctlopts.RunControlOpts(opts)
        q_mpi = opts.get_MPI(j)
        vopts = opts["vulcan"]
    vopts = vopts.__class__(vopts)
    # Apply other options
    vopts.set_opts(kw)
    # Start the command with the executable name
    cmdi = ["vulcan"]
    # Parallel environments require the two positional arguments
    if q_mpi:
        # Number of processors
        nproc = vopts.get_opt("nproc", j=j)
        # Fall back to "RunControl" settings and environment
        if nproc is None:
            nproc = get_nproc(opts, j=j)
        cmdi.append(str(int(nproc)))
        # Host file name; fall back to the PBS nodefile if present
        fhost = vopts.get_opt("hostfile", j=j)
        if not fhost:
            fhost = os.environ.get("PBS_NODEFILE")
        # A dummy value is acceptable if the site doesn't need one
        cmdi.append(fhost if fhost else "NULL")
    # Combine the p/s/g/r execution flags into one token
    flags = ''.join(
        cc for kk, cc in _VULCAN_FLAG_ORDER if vopts.get_opt(kk, j=j)
    )
    if flags:
        cmdi.append('-' + flags)
    # Name of the VULCAN input file
    finp = vopts.get_opt("inpfile", j=j)
    if finp is not None:
        cmdi.append(str(getel(finp, j)))
    # Optional screen output file
    fout = vopts.get_opt("outfile", j=j)
    if fout is not None:
        cmdi.append(str(getel(fout, j)))
    # Loop through any additional command-line inputs
    for k in vopts:
        # Skip the explicitly processed options
        if k in _VULCAN_SKIP:
            continue
        # Get the value
        v = vopts.get_opt(k, j=j)
        # Check the type
        if v is True:
            # Just a flag with no value
            cmdi.append('-' + k)
        elif v is False or v is None:
            # Do not use
            pass
        else:
            # Append the option and value
            cmdi.append('-' + k)
            cmdi.append(str(getel(v, j)))
    # Output
    return cmdi
