r"""
:mod:`cape.pylch.options.chemopts`: Options for Loci/CHEM executable
=====================================================================
"""

# Local imports
from ...cfdx.options.execopts import ExecOpts


# Options for ``chem`` executable
class ChemOpts(ExecOpts):
    r"""Class for ``superlava`` settings

    :Call:
        >>> opts = SuperlavaOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`SuperlavaOpts`
            CLI options for ``superlava``
    :Versions:
        * 2024-08-01 ``@sneuhoff``: v1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "executable",
        "project",
    }

    # Types
    _opttypes = {
        "executable": str,
        "project": str,
    }

    # Defaults
    _rc = {
        "executable": "chem",
        "run": True,
        "project": "pychem",
    }

    # Descriptions
    _rst_descriptions = {
        "executable": "path to Loci/CHEM executable",
        "project": "base name for Loci/CHEM files",
    }


# Add properties
ChemOpts.add_properties(ChemOpts._optlist, prefix="chem_")
