r"""
:mod:`cape.pykes.options.runctlopts`: Run control options
==============================================================

Options interface for aspects of running a case of Kestrel. It is only
moderately modified from the template module

    :mod:`cape.cfdx.options.runctlopts`

"""

# Local imports
from .archiveopts import ArchiveOpts
from ...cfdx.options import runctlopts


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "ProjectName",
        "XMLPrefix",
    }

    # Types
    _opttypes = {
        "ProjectName": str,
        "XMLPrefix": str,
    }

    # Defaults
    _rc = {
        "ProjectName": "pykes",
        "XMLPrefix": "kestrel",
    }

    # Descriptions
    _rst_descriptions = {
        "ProjectName": "project root name, or file prefix",
        "XMLPrefix": "base name for Kestrel project XML files",
    }

    # Additional sections
    _sec_cls = {
        "Archive": ArchiveOpts,
    }


# Add properties
RunControlOpts.add_properties(RunControlOpts._optlist)

