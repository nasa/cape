r"""
:mod:`cape.cfdx.options.meshopts`: Basic options for mesh settings
==================================================================

This module provides the class :class:`MeshOpts`, which reads the
settings in the ``"Mesh"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict, BOOL_TYPES


# Class for generic mesh settings
class MeshOpts(OptionsDict):
    r"""Dictionary-based interface for *Mesh* section

    :Versions:
        * 2023-03-16 ``@ddalle``: v1.0
    """
    # No attbitues
    __slots__ = ()

    # Identifiers
    _name = "options for mesh inputs"

    # List of options
    _optlist = {
        "CopyFiles",
        "LinkFiles",
        "MeshFile",
        "TriFile",
        "LinkMesh",
    }

    # List depth
    _optlistdepth = {
        "CopyFiles": 1,
        "LinkFiles": 1,
    }
    
    # Defaults
    _rc = {
        "LinkMesh": False,
    }

    # Types
    _opttypes = {
        "MeshFile": str,
        "TriFile": str,
        "LinkMesh": BOOL_TYPES,
    }

    # Descriptions
    _rst_descriptions = {
        "CopyFiles": "file(s) to copy to run folder w/o changing file name",
        "LinkFiles": "file(s) to link into run folder w/o changing file name",
        "LinkMesh": "option to link mesh file(s) instead of copying",
        "MeshFile": "original mesh file name(s)",
        "TriFile": "original surface triangulation file(s)",
    }


# Add getters/setters
MeshOpts.add_properties(MeshOpts._optlist)
MeshOpts.add_extenders(MeshOpts._optlist)
