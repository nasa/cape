r"""
:mod:`cape.pykes.options.runctlopts`: Run control options
==============================================================

Options interface for aspects of running a case of Kestrel. It is only
moderately modified from the template module

    :mod:`cape.cfdx.options.runctlopts`

"""

# Local imports
from ...cfdx.options import runctlopts


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "ProjectName",
    }

    # Types
    _opttypes = {
        "ProjectName": str,
    }

    # Defaults
    _rc = {
        "ProjectName": "pykes",
    }

    # Descriptions
    _rst_descriptions = {
        "ProjectName": "project root name, or file prefix",
    }


# Add properties
RunControlOpts.add_properties(RunControlOpts._optlist)

