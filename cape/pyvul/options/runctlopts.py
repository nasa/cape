r"""
:mod:`cape.pyfun.options.runctlopts`: FUN3D run control options
=================================================================

Options interface for aspects of running a case of FUN3D.  The settings
are read from the ``"RunControl"`` of a JSON file, and the contents of
this section are written to ``case.json`` within each run folder.

The FUN3D-specific options include adaptation settings and command-line
arguments for both ``nodet`` and ``dual``.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
    * :mod:`cape.cfdx.options.archiveopts`
    * :mod:`cape.pyfun.options.archiveopts`
"""

# Local imports
from ...cfdx.options import runctlopts
from ...cfdx.options.execopts import ExecOpts
from ...optdict import INT_TYPES


# Class for `nodet` inputs
class VulcanOpts(ExecOpts):
    r"""Class for ``vulcan`` executable settings

    :Call:
        >>> opts = VulcanOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`VulcanOpts`
            CLI options for ``vulcan``
    """
    # Attributes
    __slots__ = ()

    # Identifiers
    _name = "CLI options for ``vulcan``, the main VULCAN-CFD executable"

    # Accepted options
    _optlist = (
        "nproc",
        "inpfile",
    )

    # Types
    _opttypes = {
        "nproc": INT_TYPES,
        "inpfile": str,
    }

    # Defaults
    _rc = {
        "inpfile": "pyvul.inp",
    }

    # Descriptions
    _rst_descriptions = {
        "nproc": "number of CPU 'procs' to use in run",
        "inpfile": "name of VULCAN input file",
    }


# Add properties
VulcanOpts.add_properties(VulcanOpts._optlist, prefix="vulcan_")


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    r"""VULCAN-specific "RunControl" options interface

    :Call:
        >>> opts = RunControl(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of "RunControl" settings
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    """
   # === Class attributes ===
    # Disallow other attributes
    __slots__ = tuple()

    # Names of allowed settings
    _optlist = (
        "vulcan",
    )

    # Option types
    _opttypes = {}

    # Option values
    _optvals = {}

    # Default values
    _rc = {}

    # Descriptions
    _rst_descriptions = {}

    # Additional sections
    _sec_cls = {
        "vulcan": VulcanOpts,
    }


# Create properties
# RunControlOpts.add_properties(())
# Upgrade subsections
RunControlOpts.promote_sections()
