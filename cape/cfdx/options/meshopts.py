r"""
Basic options for mesh settings

This module provides the class :class:`MeshOpts`, which reads the
settings in the ``"Mesh"`` section of the main CAPE control file.

"""


# Local imports
from ...optdict import OptionsDict


# Class for generic mesh settings
class MeshOpts(OptionsDict):
    # No attbitues
    __slots__ = ()

    # List of options
    _optlist = {
        "MeshFile",
        "TriFile",
    }

    # Types
    _optlist = {
        "MeshFile": str,
        "TriFile": str,
    }

    # Descriptions
    _rst_descriptions = {
        "MeshFile": "original mesh file name(s)",
        "TriFile": "original surface triangulation file(s)",
    }


# Add getters/setters
MeshOpts.add_properties(MeshOpts._optlist)
MeshOpts.add_extenders(MeshOpts._optlist)
