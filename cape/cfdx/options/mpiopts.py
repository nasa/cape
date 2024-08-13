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
from .util import ExecOpts


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
        * 2016-04-04 ``@ddalle``: Version 1.0 (:class:`aflr3`)
        * 2022-10-14 ``@ddalle``: Version 2.0
    """
    __slots__ = tuple()

    _optlist = {
        "executable",
        "np",
        "flags",
    }

    _opttypes = {
        "executable": str,
        "flags": dict,
        "np": INT_TYPES,
    }

    _rc = {
        "executable": "mpiexec",
    }

    # Descriptions
    _rst_descriptions = {
        "executable": "executable to launch MPI",
        "np": "explicit number of MPI processes",
        "flags": "AFLR3 options using ``-flag val`` format",
    }


# Add all properties
MPIOpts.add_properties(MPIOpts._optlist, prefix="mpi_")

