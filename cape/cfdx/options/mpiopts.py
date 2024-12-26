r"""
:mod:`cape.cfdx.options.mpiopts`: MPI command-line launch options
====================================================================

This module provides a class to access command-line options to
explicitly launch MPI, usually using ``mpiexec`` or ``mpirun``. It is
used to form command prefixes like ``mpiexec -np 8`` and add other
options. Generic command-line options can be added using the ``"flags"``
entry in this section, which is a :class:`dict`.

    * :func:`MPIOpts.get_mpi_flags`: opts with ``-blr 1.2`` format
"""

# Local imports
from ...optdict import INT_TYPES
from .execopts import ExecOpts


# Resource limits class
class MPIOpts(ExecOpts):
    r"""Class for MPI command-line settings

    :Call:
        >>> opts = MPIOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of MPI command-line options
    :Outputs:
        *opts*: :class:`MPIOpts`
            MPI options interface
    :Versions:
        * 2022-10-14 ``@ddalle``: v1.0
    """
    __slots__ = tuple()

    _name = "options for MPI executable and command-line options"

    _optlist = {
        "args",
        "executable",
        "np",
        "perhost",
        "flags",
    }

    _optmap = {
        "nhost": "perhost",
    }

    _opttypes = {
        "args": str,
        "executable": str,
        "flags": dict,
        "np": INT_TYPES,
        "perhost": INT_TYPES,
    }

    _optlistdepth = {
        "args": 1,
    }

    _rc = {
        "executable": "mpiexec",
    }

    # Descriptions
    _rst_descriptions = {
        "executable": "executable to launch MPI",
        "nhost": "explicit number of MPI processes (gpu)",
        "np": "explicit number of MPI processes",
        "flags": "options to ``mpiexec`` using ``-flag val`` format",
    }


# Add all properties
MPIOpts.add_properties(MPIOpts._optlist, prefix="mpi_")

