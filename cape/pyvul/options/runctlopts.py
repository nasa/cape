r"""
:mod:`cape.pyvul.options.runctlopts`: VULCAN run control options
==================================================================

Options interface for aspects of running a case of VULCAN-CFD.  The
settings are read from the ``"RunControl"`` of a JSON file, and the
contents of this section are written to ``case.json`` within each run
folder.

The VULCAN-specific options include the ``"ProjectRootname"`` CAPE
setting (there is no such setting in the VULCAN input file) and the
command-line arguments for the main ``vulcan`` executable.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
"""

# Local imports
from ...cfdx.options import runctlopts
from ...cfdx.options.execopts import ExecOpts
from ...optdict import BOOL_TYPES, INT_TYPES


# Class for `vulcan` inputs
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
        "hostfile",
        "inpfile",
        "outfile",
        "pre",
        "solve",
        "post",
        "recompose",
    )

    # Types
    _opttypes = {
        "nproc": INT_TYPES,
        "hostfile": str,
        "inpfile": str,
        "outfile": str,
        "pre": BOOL_TYPES,
        "solve": BOOL_TYPES,
        "post": BOOL_TYPES,
        "recompose": BOOL_TYPES,
    }

    # Defaults
    _rc = {
        "inpfile": "vulcan.inp",
        "pre": True,
        "solve": True,
    }

    # Descriptions
    _rst_descriptions = {
        "nproc": "number of CPU 'procs' to use in run",
        "hostfile": "MPI hostfile name (falls back to $PBS_NODEFILE)",
        "inpfile": "name of VULCAN input file",
        "outfile": "name of VULCAN screen output file",
        "pre": "execute preprocessor steps (``-p``)",
        "solve": "execute flow solver (``-s``)",
        "post": "execute postprocessor (``-g``)",
        "recompose": "recompose structured grid data (``-r``)",
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
        "ProjectRootname",
        "vulcan",
    )

    # Option types
    _opttypes = {
        "ProjectRootname": str,
    }

    # Option values
    _optvals = {}

    # Default values
    _rc = {
        "ProjectRootname": "vulcan",
    }

    # Descriptions
    _rst_descriptions = {
        "ProjectRootname": "root name for project output files",
    }

    # Additional sections
    _sec_cls = {
        "vulcan": VulcanOpts,
    }


# Create properties
RunControlOpts.add_properties(("ProjectRootname",))
# Upgrade subsections
RunControlOpts.promote_sections()
