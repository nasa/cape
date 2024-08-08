r"""
:mod:`cape.cfdx.options.aflr3opts`: AFLR3 mesh generation options
====================================================================

This module provides a class to access command-line options to the AFLR3
mesh-generation program. It is specified in the ``"RunControl"`` section
for modules that utilize the solver, which includes FUN3D.

The options in this module are among the command-line options to AFLR3.
Other AFLR3 options that do not have specific methods defined in the
:class:`AFLR3Opts` options class can be accessed using two generic
functions:

    * :func:`AFLR3Opts.get_aflr3_flags`: opts with ``-blr 1.2`` format
    * :func:`AFLR3Opts.get_aflr3_keys`: options with ``cdfs=7.5`` format
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

